"""Sequential, resumable media ingestion and transcription pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..config import AppConfig
from ..domain.models import (
    AudioChunk,
    DegenerateOutputError,
    PreparedAudio,
    Segment,
    SegmentationSettings,
    Transcript,
    TranscriptionError,
    TransientBackendError,
)
from ..domain.ports import MediaProcessor, TranscriptExporter, TranscriptionBackend
from .catalog import refresh_source, version_summary
from .events import NULL_EVENTS
from .layout import (
    LayoutError,
    find_source_folder,
    find_version_folder,
    parse_version_folder,
    scope_label,
    source_folder_name,
    source_key,
    version_folder_name,
)
from .progress import ProgressTracker
from .reporting import render_run_report, sum_stages, timings_line
from .state import atomic_json, read_json

MEDIA_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
    ".wmv",
}

NO_SPEECH_WARNING = "No speech detected"
EXECUTION_SCHEMA_VERSION = 3
ARCHIVE_DURATION_TOLERANCE = 0.05  # ffprobe rounds to ms; codecs pad frames


def _segment_dict(segment: Segment) -> dict:
    return asdict(segment)


def _segment_from_dict(data: dict) -> Segment:
    return Segment(
        start=float(data["start"]),
        end=float(data["end"]),
        text=str(data["text"]),
        speaker=data.get("speaker"),
        timing_source=data.get("timing_source", "chunk"),
    )


class TranscriptionPipeline:
    def __init__(
        self,
        config: AppConfig,
        processor: MediaProcessor,
        backend: TranscriptionBackend | None = None,
        exporter: TranscriptExporter | None = None,
        *,
        status: Callable[[str], None] = print,
        events=None,
    ):
        self.config = config
        self.processor = processor
        self.backend = backend
        self.exporter = exporter
        self._status_out = status
        self.events = events if events is not None else NULL_EVENTS
        self._total_chunks = 0
        self._job: dict | None = None
        self._job_id: str | None = None
        self._job_open = False
        self._source_name: str | None = None
        self._queue_total = 0
        self._queue_index = 0
        self._tracker: ProgressTracker | None = None
        self._clock: Callable[[], float] = time.monotonic

    def attach(self, *, events=None, status: Callable[[str], None] | None = None) -> None:
        """Route events and plain status lines (used by callers that own the run session)."""
        if events is not None:
            self.events = events
        if status is not None:
            self._status_out = status

    def status(self, message: str) -> None:
        """Print a status line (unchanged text) and publish it as a ``log`` event."""
        self._status_out(message)
        self.events.emit(
            "log", {"level": _status_level(message), "message": message}, job_id=self._job_id
        )

    def _emit(self, type_: str, **data) -> None:
        self.events.emit(type_, data, job_id=self._job_id)

    def _stage_finished(self, name: str, seconds: float | None, **extra) -> None:
        self._emit(
            "stage.finished",
            stage=name,
            seconds=None if seconds is None else round(seconds, 3),
            **extra,
        )

    def discover(self) -> list[Path]:
        root = self.config.input_dir
        if not root.exists():
            return []
        excluded = {".git", "process", "output"}
        found = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part.startswith(".") or part in excluded for part in relative.parts):
                continue
            found.append(path)
        return sorted(found, key=lambda p: str(p).casefold())

    def _identity(self, source: Path, fingerprint: str) -> tuple[str, dict]:
        """Return ``("<source-folder>/<version-folder>", identity)``.

        The job is identified by the media bytes (source key) and the configuration
        fingerprint, never by the path. An existing version folder is found by its unique
        fingerprint suffix; otherwise a new name is derived with the current UTC time.
        """
        stat = source.stat()
        key = source_key(source)
        roots = [self.config.process_dir, self.config.output_dir, self.config.processed_dir]
        folder = find_source_folder(roots, key) or source_folder_name(source.name, key)
        version = find_version_folder(
            [self.config.process_dir, self.config.output_dir], folder, fingerprint
        ) or version_folder_name(
            datetime.now(UTC), self.config.model, self.config.max_duration, fingerprint
        )
        identity = {"source_key": key, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return f"{folder}/{version}", identity

    def _job_dirs(self, job_id: str) -> tuple[Path, Path]:
        folder, version = job_id.split("/", 1)
        return (
            self.config.process_dir / folder / version,
            self.config.output_dir / folder / version,
        )

    def _fingerprint(self) -> str:
        config = self.config
        data = {
            "backend": config.backend,
            "model": config.model,
            "model_path": _path_signature(config.model_path),
            "projector_path": _path_signature(config.projector_path),
            "device": config.device,
            "language": config.language,
            "chunk_seconds": config.chunk_seconds,
            "sample_rate": config.sample_rate,
            "max_duration": config.max_duration,
            "segmentation": {
                "vad": config.vad,
                "vad_model_path": (
                    _path_signature(config.vad_model_path) if config.vad == "silero" else None
                ),
                "threshold": config.vad_threshold,
                "min_speech_seconds": config.vad_min_speech_seconds,
                "min_silence_seconds": config.vad_min_silence_seconds,
                "speech_pad_seconds": config.vad_speech_pad_seconds,
                "max_merge_gap_seconds": config.vad_max_merge_gap_seconds,
                "energy_margin_db": config.vad_energy_margin_db,
            },
            "fallback": {
                "temperatures": list(config.fallback_temperatures),
                "split_on_failure": config.split_on_failure,
                **({} if config.context_free_fallback else {"context_free_fallback": False}),
            },
            "backend_settings": (
                {
                    "base_url": config.base_url,
                    "timeout": config.timeout,
                    "max_tokens": config.max_tokens,
                    "prompt": config.prompt,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "response_mode": config.response_mode,
                    "min_tokens": config.min_tokens,
                    "tokens_per_second": config.tokens_per_second,
                    "compression_ratio_threshold": config.compression_ratio_threshold,
                    "repeat_penalty": config.repeat_penalty,
                    "dry_multiplier": config.dry_multiplier,
                    "force_language": config.force_language,
                    # Present only when enabled, so a run without logprobs keeps the
                    # fingerprint it had before this setting existed.
                    **({"collect_logprobs": True} if config.collect_logprobs else {}),
                    # Batched GPU execution may change greedy output slightly, so concurrent
                    # runs are a distinct version; 1 is omitted to keep earlier fingerprints.
                    **(
                        {"parallel_requests": config.parallel_requests}
                        if config.parallel_requests > 1
                        else {}
                    ),
                }
                if config.backend == "llamacpp"
                else None
            ),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()

    def _segmentation_settings(self) -> SegmentationSettings:
        config = self.config
        return SegmentationSettings(
            detector=config.vad,
            max_chunk_seconds=config.chunk_seconds,
            threshold=config.vad_threshold,
            min_speech_seconds=config.vad_min_speech_seconds,
            min_silence_seconds=config.vad_min_silence_seconds,
            speech_pad_seconds=config.vad_speech_pad_seconds,
            max_merge_gap_seconds=config.vad_max_merge_gap_seconds,
            energy_margin_db=config.vad_energy_margin_db,
        )

    def _load_prepared(self, job_dir: Path, fingerprint: str) -> PreparedAudio | None:
        manifest = read_json(job_dir / "prepared.json")
        if not isinstance(manifest, dict) or manifest.get("fingerprint") != fingerprint:
            return None
        try:
            items = manifest["chunks"]
            if not isinstance(items, list):
                return None
            duration = float(manifest["duration"])
            speech_seconds = float(manifest["speech_seconds"])
            segmentation = manifest["segmentation"]
            if (
                not math.isfinite(duration)
                or duration <= 0
                or not math.isfinite(speech_seconds)
                or speech_seconds < 0
                or not isinstance(segmentation, dict)
            ):
                return None
            chunks = []
            previous_end = 0.0
            for expected_index, item in enumerate(items):
                if not isinstance(item, dict):
                    return None
                path = Path(item["path"]).resolve()
                path.relative_to(job_dir.resolve())
                if not path.is_file():
                    return None
                start, end, index = float(item["start"]), float(item["end"]), int(item["index"])
                part = item.get("part", "")
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or start < previous_end - 0.000001
                    or end <= start
                    or index != expected_index
                    or part != ""
                ):
                    return None
                chunks.append(AudioChunk(path, start, end, index))
                previous_end = end
            return PreparedAudio(tuple(chunks), duration, speech_seconds, segmentation)
        except (KeyError, TypeError, ValueError, OSError):
            return None

    def run(
        self,
        sources: Iterable[Path] | None = None,
        *,
        close_backend: bool = True,
        limit: int | None = None,
    ) -> list[dict]:
        source_list = list(self.discover() if sources is None else sources)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be at least 1")
            source_list = source_list[:limit]
        self.config.process_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.input_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self._queue_total = len(source_list)
        self._queue_index = 0
        fingerprint = self._fingerprint()
        results: list[dict | None] = [None] * len(source_list)
        eligible = []
        if self.config.max_file_size is not None:
            for index, source in enumerate(source_list):
                try:
                    size = source.stat().st_size
                except OSError as exc:
                    error = f"Could not inspect source file: {exc}"
                    self.status(f"Failed {source.name}: {error}")
                    self._reject_job(source.name, error)
                    results[index] = {
                        "source": str(source.resolve()),
                        "status": "failed",
                        "error": error,
                    }
                    continue
                if size > self.config.max_file_size:
                    error = "Input file exceeds max_file_size"
                    self.status(f"Skipped {source.name}: {error}")
                    self._record_rejected(source, fingerprint, error)
                    self._reject_job(source.name, error)
                    results[index] = {
                        "source": str(source.resolve()),
                        "status": "failed",
                        "error": error,
                    }
                    continue
                eligible.append((index, source))
        else:
            eligible = list(enumerate(source_list))
        eligible_sources = [source for _, source in eligible]
        if not self.config.prepare_only and eligible_sources:
            if self.backend is None:
                raise RuntimeError("The selected transcription backend is unavailable")
            # Fail before probing or extracting potentially large media.
            self.backend.check()

        try:
            for index, source in eligible:
                try:
                    result = self._run_one(source, fingerprint)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self.status(f"Failed {source.name}: {error}")
                    result = {"status": "failed", "error": error}
                    self._close_job_after_error(source, result)
                result.setdefault("source", str(source.resolve()))
                results[index] = result
        finally:
            if close_backend and self.backend is not None:
                self.backend.close()
        return [result for result in results if result is not None]

    def _record_rejected(self, source: Path, fingerprint: str, message: str) -> None:
        try:
            job_id, identity = self._identity(source, fingerprint)
        except (OSError, LayoutError):
            return
        work_dir, _ = self._job_dirs(job_id)
        now = _utc_now()
        failure = {
            "job_id": job_id,
            "source_name": source.name,
            "status": "failed",
            "error": message,
            "updated_at": now,
        }
        atomic_json(work_dir / "failure.json", failure)
        atomic_json(
            work_dir / "metadata.json",
            {
                "job_id": job_id,
                "source_key": identity["source_key"],
                "identity": identity,
                "config_fingerprint": fingerprint,
                "source_name": source.name,
                "last_seen_path": str(source.resolve()),
                "created_at": now,
                "model": self.config.model,
                "scope": _scope(self.config.max_duration),
                "status": "failed",
                "error": message,
                "updated_at": now,
            },
        )
        self._refresh_layout(job_id.split("/", 1)[0])

    def _archive_input(
        self, source: Path, source_folder: str, guard: dict
    ) -> tuple[str | None, str | None]:
        input_root = self.config.input_dir.resolve()
        try:
            source_parent = source.parent.resolve()
        except OSError as exc:
            return None, f"Could not resolve input location for archiving: {exc}"
        if source_parent != input_root:
            return None, None
        if source.is_symlink():
            return None, "Queued symbolic links are not archived automatically."
        try:
            if _source_identity(source) != guard:
                raise RuntimeError("Source changed after transcription; archive was skipped")
            archive_dir = self.config.processed_dir / source_folder
            archive_dir.mkdir(parents=True, exist_ok=True)
            destination = archive_dir / source.name
            if destination.is_symlink():
                raise FileExistsError(f"Archive destination is a symbolic link: {destination.name}")
            if destination.exists():
                if not destination.is_file():
                    raise FileExistsError(
                        f"Archive destination is not a regular file: {destination.name}"
                    )
                if not _same_file_bytes(source, destination):
                    raise FileExistsError(
                        f"Archive destination already contains different data: {destination.name}"
                    )
                if _source_identity(source) != guard:
                    raise RuntimeError("Source changed while checking an existing archive copy")
                source.unlink()
                return str(destination.resolve()), None
            if os.stat(source).st_dev != os.stat(archive_dir).st_dev:
                raise OSError(
                    "Input and processed directories are on different volumes; atomic move is unavailable"
                )
            os.rename(source, destination)
            return str(destination.resolve()), None
        except OSError as exc:
            return None, f"Could not archive input: {exc}"
        except RuntimeError as exc:
            return None, str(exc)

    def _run_one(self, source: Path, fingerprint: str) -> dict:
        self._job = None
        self._job_id = None
        self._job_open = False
        self._source_name = source.name
        self._tracker = None
        result = self._run_job(source, fingerprint)
        self._finalize_run(result)
        self._job_id = None
        return result

    # ------------------------------------------------------------------ job events
    def _job_started(self, source_name: str, job_id: str | None, version: str | None) -> None:
        self._queue_index += 1
        self._job_open = True
        self.events.emit(
            "job.started",
            {
                "source": source_name,
                "version": version,
                "scope": scope_label(self.config.max_duration),
                "index": self._queue_index,
                "total": max(self._queue_total, self._queue_index),
            },
            job_id=job_id,
        )

    def _reject_job(self, source_name: str, error: str) -> None:
        """A source refused before any job state exists: still a started/finished pair."""
        self._job_started(source_name, None, None)
        self.events.emit(
            "job.finished",
            {"source": source_name, "status": "failed", "error": _short(error)},
        )
        self._job_open = False

    def _close_job_after_error(self, source: Path, result: dict) -> None:
        job_id = self._job_id
        if not self._job_open:
            self._job_started(source.name, job_id, None)
        self._emit_job_finished(result, None, None, 0, source.name)
        self._job = None
        self._job_id = None

    def _emit_job_finished(
        self,
        result: dict,
        run: dict | None,
        checkpoint: dict | None,
        total_chunks: int,
        source_name: str | None,
        duration: float | None = None,
    ) -> None:
        job_id = result.get("job_id") or self._job_id
        counts = confidence = None
        if isinstance(checkpoint, dict) and duration:
            try:
                summary = _execution_summary(checkpoint, total_chunks, duration)
                counts = {
                    "total_chunks": total_chunks,
                    "ok": summary["chunks_ok"],
                    "no_speech": summary["chunks_no_speech"],
                    "failed": summary["chunks_failed"],
                    "fallbacks": summary["temperature_fallbacks_used"],
                    "splits": summary["splits_used"],
                    "tokens": summary["completion_tokens"]["sum"],
                }
                if isinstance(summary.get("confidence"), dict):
                    confidence = summary["confidence"].get("mean")
            except Exception:
                counts = confidence = None
        timings = None
        if run is not None:
            timings = {
                "wall_seconds": run.get("wall_seconds"),
                "stages": dict(run.get("stages_seconds") or {}),
            }
        artifacts: list[str] = []
        if isinstance(job_id, str) and result.get("status") not in {"prepared"}:
            base = self.config.output_dir / job_id
            for folder in ("", "intermediate/"):
                for name in (
                    "transcript.txt",
                    "transcript.md",
                    "transcript.srt",
                    "transcript.vtt",
                    "transcript.json",
                    "report.json",
                    "run-report.md",
                ):
                    try:
                        found = (base / (folder + name)).is_file()
                    except OSError:
                        found = False
                    if found:
                        artifacts.append(f"output/{job_id}/{folder}{name}")
                if artifacts and folder == "":
                    break
        data = {
            "source": source_name,
            "status": result.get("status"),
            "counts": counts,
            "timings": timings,
            "confidence": confidence,
            "artifacts": artifacts,
            "archived": bool(result.get("archived_source_path")),
        }
        if result.get("error"):
            data["error"] = _short(str(result["error"]))
        self.events.emit("job.finished", data, job_id=job_id if isinstance(job_id, str) else None)
        self._job_open = False

    def _refresh_layout(self, source_folder: str) -> None:
        """Update source.json and catalog.json; bookkeeping must never fail a job."""
        started = time.monotonic()
        try:
            refresh_source(self.config, source_folder)
        except Exception as exc:  # derived files can always be rebuilt with `catalog`
            self.status(f"Could not update the catalog for {source_folder}: {exc}")
        if self._job is not None:
            stages = self._job["stages"]
            stages["catalog"] = round(stages.get("catalog", 0.0) + time.monotonic() - started, 3)

    def _current_stages(self) -> dict | None:
        """Cumulative stage timings over earlier runs plus the run in progress."""
        job = self._job
        if job is None:
            return None
        earlier = [r for r in job["state"].get("runs", [])[:-1] if isinstance(r, dict)]
        return sum_stages([*earlier, {"stages_seconds": dict(job["stages"])}])

    def _finalize_run(self, result: dict) -> None:
        """Close the run record, write the run report, refresh the catalog, re-render.

        The catalog lists the report, so it is refreshed after the first render; its time is
        then recorded and the report is rendered again so no stage is shown as zero.
        """
        job = self._job
        job_id = result.get("job_id")
        source_folder = (
            job_id.split("/", 1)[0] if isinstance(job_id, str) and "/" in job_id else None
        )
        if job is not None:
            run = job["run"]
            run["ended_at"] = _utc_now()
            run["wall_seconds"] = round(time.monotonic() - job["started"], 3)
            run["status"] = result.get("status")
            run["stages_seconds"] = dict(job["stages"])
            self._write_run_report(job, result)
        if source_folder is not None:
            self._refresh_layout(source_folder)  # adds to the job's "catalog" stage
        if job is None:
            self._emit_job_finished(result, None, None, 0, getattr(self, "_source_name", None))
            return
        run = job["run"]
        run["stages_seconds"] = dict(job["stages"])
        run["wall_seconds"] = round(time.monotonic() - job["started"], 3)
        self._write_run_report(job, result)
        self._job = None
        if result.get("status") in {"completed", "incomplete", "failed"}:
            self.status(timings_line(run["wall_seconds"], run["stages_seconds"]))
        if "catalog" in run["stages_seconds"]:
            self._stage_finished("catalog", run["stages_seconds"]["catalog"])
        self._emit_job_finished(
            result,
            run,
            job.get("checkpoint"),
            job["total_chunks"],
            job["state"].get("source_name"),
            job.get("duration"),
        )

    def _write_run_report(self, job: dict, result: dict) -> None:
        state = job["state"]
        work_dir, output_dir = self._job_dirs(state["job_id"])
        try:
            atomic_json(work_dir / "metadata.json", state)
            cumulative = self._current_stages()
            execution = None
            checkpoint = job.get("checkpoint")
            if isinstance(checkpoint, dict) and job.get("duration"):
                execution = _execution_summary(
                    checkpoint,
                    job["total_chunks"],
                    job["duration"],
                    cumulative,
                    self.config.parallel_requests,
                )
            completion = version_summary(self.config, *state["job_id"].split("/", 1))[
                "completion_percent"
            ]
            text = render_run_report(state, execution, completion_percent=completion)
            targets = [output_dir / "intermediate" / "run-report.md"]
            if result.get("status") == "completed":
                targets.append(output_dir / "run-report.md")
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_text(target, text)
        except Exception as exc:
            self.status(f"Could not write the run report: {exc}")

    def _run_job(self, source: Path, fingerprint: str) -> dict:
        job_id, identity = self._identity(source, fingerprint)
        guard = _source_identity(source)
        work_dir, output_dir = self._job_dirs(job_id)
        source_folder, version_folder = job_id.split("/", 1)
        self._job_id = job_id
        self._job_started(source.name, job_id, version_folder)
        metadata_path = work_dir / "metadata.json"
        metadata = read_json(metadata_path, {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        required_artifacts = tuple(
            output_dir / filename
            for filename in (
                "transcript.json",
                "transcript.txt",
                "transcript.md",
                "transcript.srt",
                "transcript.vtt",
                "report.json",
            )
        )
        if (
            not self.config.force
            and metadata.get("status") == "completed"
            and metadata.get("config_fingerprint") == fingerprint
            and all(path.is_file() for path in required_artifacts)
        ):
            self.status(f"Skipped {source.name}: already completed")
            result = {"job_id": job_id, "source": str(source.resolve()), "status": "skipped"}
            backend = metadata.get("backend", self.config.backend)
            if self.config.archive_inputs and backend != "mock":
                archive_path = None
                warning = _archive_duration_warning(metadata)
                if warning is None:
                    archive_path, warning = self._archive_input(source, source_folder, guard)
                if archive_path is not None:
                    metadata["archived_source_path"] = archive_path
                    metadata.pop("archive_warning", None)
                    atomic_json(metadata_path, metadata)
                    result["archived_source_path"] = archive_path
                elif warning:
                    metadata["archive_warning"] = warning
                    atomic_json(metadata_path, metadata)
                    result["archive_warning"] = warning
            return result

        runs = metadata.get("runs") if isinstance(metadata.get("runs"), list) else []
        created = metadata.get("created_at")
        if not isinstance(created, str):
            parsed = parse_version_folder(version_folder)
            # A brand-new version records the exact time; a legacy one only knows its folder.
            created = parsed["created"].isoformat() if parsed and metadata else _utc_now()
        run = {
            "started_at": _utc_now(),
            "ended_at": None,
            "wall_seconds": None,
            "status": "processing",
            "stages_seconds": {},
        }
        if getattr(self.events, "run_id", None):
            run["run_id"] = self.events.run_id
        state = {
            "job_id": job_id,
            "source_folder": source_folder,
            "version_folder": version_folder,
            "source_key": identity["source_key"],
            "identity": identity,
            "config_fingerprint": fingerprint,
            "source_name": source.name,
            "last_seen_path": str(source.resolve()),
            "created_at": created,
            "model": self.config.model,
            "backend": getattr(self.backend, "name", self.config.backend),
            "language": self.config.language,
            "response_mode": (
                self.config.response_mode if self.config.backend == "llamacpp" else None
            ),
            "scope": _scope(self.config.max_duration),
            "status": "processing",
            "updated_at": _utc_now(),
            "runs": [*[r for r in runs if isinstance(r, dict)], run],
        }
        if self.config.backend == "llamacpp":
            state["inference_configuration"] = self._inference_configuration()
        self._job = {
            "state": state,
            "run": run,
            "stages": {},
            "started": time.monotonic(),
            "checkpoint": None,
            "duration": None,
            "total_chunks": 0,
        }
        atomic_json(metadata_path, state)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_layout(source_folder)
        segments: list[Segment] = []
        intermediate: dict | None = None
        try:
            stage_started = time.monotonic()
            self._emit("stage.started", stage="probe")
            info = self.processor.probe(source)
            probe_seconds = time.monotonic() - stage_started
            self._stage_finished("probe", probe_seconds)
            prepared = None if self.config.force else self._load_prepared(work_dir, fingerprint)
            rebuilt_chunks = prepared is None
            prep_seconds: float | None = None
            if prepared is None:
                prep_started = time.monotonic()
                self._emit("stage.started", stage="preparation")
                prepared = self.processor.prepare(
                    source,
                    work_dir,
                    sample_rate=self.config.sample_rate,
                    segmentation=self._segmentation_settings(),
                    max_duration=self.config.max_duration,
                )
                prep_seconds = time.monotonic() - prep_started
                duration = _validate_prepared(prepared, work_dir, self.config.max_duration)
                if _source_identity(source) != guard:
                    raise RuntimeError("Source media changed while audio preparation was running")
                atomic_json(
                    work_dir / "prepared.json",
                    {
                        "fingerprint": fingerprint,
                        "source_duration": info.duration,
                        "duration": duration,
                        "speech_seconds": prepared.speech_seconds,
                        "segmentation": prepared.segmentation,
                        "chunks": [
                            {
                                "path": str(c.path.resolve()),
                                "start": c.start,
                                "end": c.end,
                                "index": c.index,
                                "part": c.part,
                            }
                            for c in prepared.chunks
                        ],
                    },
                )
            else:
                duration = prepared.duration
            stages = self._job["stages"]
            stages["probe"] = round(probe_seconds, 3)
            for key, value in (prepared.timings or {}).items():
                stages[key] = round(float(value), 3)
            stages["preparation"] = round(probe_seconds + (prep_seconds or 0.0), 3)
            for key in ("audio_extraction", "speech_detection", "chunk_slicing"):
                self._stage_finished(
                    key,
                    stages.get(key) if prep_seconds is not None else None,
                    reused=prep_seconds is None,
                )
            self._stage_finished("preparation", stages["preparation"], reused=prep_seconds is None)
            chunks = list(prepared.chunks)
            self._total_chunks = len(chunks)
            self._job["duration"] = duration
            self._job["total_chunks"] = len(chunks)
            self.status(_prepared_summary(source.name, prepared, duration, prep_seconds))
            lengths = [chunk.end - chunk.start for chunk in chunks]
            self._emit(
                "job.prepared",
                analysed_seconds=round(duration, 3),
                speech_seconds=round(prepared.speech_seconds, 3),
                chunk_count=len(chunks),
                chunk_seconds_min=round(min(lengths), 2) if lengths else None,
                chunk_seconds_avg=round(sum(lengths) / len(lengths), 2) if lengths else None,
                chunk_seconds_max=round(max(lengths), 2) if lengths else None,
                detector=prepared.segmentation.get("detector"),
                reused=prep_seconds is None,
            )
            audio_metadata = {
                "sample_rate": self.config.sample_rate,
                "channels": 1,
                "codec": "pcm_s16le",
                "duration_seconds": duration,
                "speech_seconds": prepared.speech_seconds,
                "chunk_count": len(chunks),
                "segmentation": prepared.segmentation,
            }
            source_metadata = {
                "source_name": source.name,
                "size_bytes": identity["size"],
                "mtime_ns": identity["mtime_ns"],
                "duration_seconds": info.duration,
            }
            state.update(
                status="prepared",
                chunk_count=len(chunks),
                source_metadata=source_metadata,
                audio_metadata=audio_metadata,
                media_info={
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "codec": info.codec,
                    "duration_seconds": info.duration,
                },
                updated_at=_utc_now(),
            )
            atomic_json(metadata_path, state)
            self._refresh_layout(source_folder)
            if self.config.prepare_only:
                self.status(f"Prepared {source.name} ({len(chunks)} chunks)")
                return {"job_id": job_id, "source": str(source.resolve()), "status": "prepared"}
            if self.backend is None or self.exporter is None:
                raise RuntimeError("Backend or transcript exporter is unavailable")

            base_warnings = [] if chunks else [NO_SPEECH_WARNING]
            intermediate = {
                "job_id": job_id,
                "backend": getattr(self.backend, "name", self.config.backend),
                "model": self.config.model,
                "duration": duration,
                "source_metadata": source_metadata,
                "audio_metadata": audio_metadata,
                "total_chunks": len(chunks),
                "warnings": base_warnings,
            }

            checkpoint_path = work_dir / "checkpoint.json"
            checkpoint = (
                {}
                if self.config.force or rebuilt_chunks
                else (read_json(checkpoint_path, {}) or {})
            )
            if not _valid_checkpoint(checkpoint, fingerprint, chunks):
                checkpoint = {}
            if checkpoint.get("fingerprint") != fingerprint:
                checkpoint = {"fingerprint": fingerprint, "chunks": {}}
            done = checkpoint.setdefault("chunks", {})
            failed = checkpoint.setdefault("failed", {})
            self._job["checkpoint"] = checkpoint
            run["resumed_chunks"] = len(done)
            inference_started = time.monotonic()
            if done:
                self.status(
                    f"Resuming {source.name}: {len(done)} of {len(chunks)} chunks "
                    "already done in the checkpoint"
                )
            self._emit("stage.started", stage="inference")
            self._transcribe_pending(
                chunks,
                work_dir,
                checkpoint,
                checkpoint_path,
                segments,
                intermediate,
                inference_started,
            )
            self._job["stages"]["inference"] = round(time.monotonic() - inference_started, 3)
            self._stage_finished("inference", self._job["stages"]["inference"])

            if failed:
                failed_indices = sorted(int(key) for key in failed)
                message = (
                    f"{len(failed_indices)} of {len(chunks)} chunk(s) could not be transcribed: "
                    + ", ".join(str(index) for index in failed_indices)
                )
                self._write_intermediate(
                    intermediate,
                    segments,
                    checkpoint=checkpoint,
                    status="incomplete",
                    error=message,
                )
                state.update(
                    status="incomplete",
                    chunk_count=len(chunks),
                    segment_count=len(segments),
                    failed_chunks=failed_indices,
                    error=message,
                    updated_at=_utc_now(),
                )
                atomic_json(metadata_path, state)
                self.status(self._execution_line(checkpoint, len(chunks), duration))
                self.status(f"Incomplete {source.name}: {message}")
                return {
                    "job_id": job_id,
                    "source": str(source.resolve()),
                    "status": "incomplete",
                    "failed_chunks": failed_indices,
                    "error": message,
                }

            transcript = Transcript(
                job_id=job_id,
                backend=getattr(self.backend, "name", self.config.backend),
                model=self.config.model,
                language=self.config.language,
                duration=duration,
                segments=sorted(segments, key=lambda segment: (segment.start, segment.end)),
                warnings=list(base_warnings),
                source_metadata=source_metadata,
                audio_metadata=audio_metadata,
            )
            if _source_identity(source) != guard:
                raise RuntimeError("Source media changed while transcription was running")
            export_started = time.monotonic()
            self._emit("stage.started", stage="export")
            self.exporter.export(transcript, output_dir)
            self._write_intermediate(
                intermediate, segments, checkpoint=checkpoint, status="completed"
            )
            self._job["stages"]["export"] = round(time.monotonic() - export_started, 3)
            self._stage_finished("export", self._job["stages"]["export"])
            inference_configuration = self._inference_configuration()
            report_path = output_dir / "report.json"
            report = read_json(report_path, {})
            if not isinstance(report, dict):
                raise RuntimeError("Transcript exporter produced an invalid report.json")
            report["execution"] = _execution_summary(
                checkpoint,
                len(chunks),
                duration,
                self._current_stages(),
                self.config.parallel_requests,
            )
            if self.config.backend == "llamacpp":
                report["inference_configuration"] = inference_configuration
            atomic_json(report_path, report)
            if self.config.backend == "llamacpp":
                transcript_path = output_dir / "transcript.json"
                transcript_payload = read_json(transcript_path, {})
                if not isinstance(transcript_payload, dict):
                    raise RuntimeError("Transcript exporter produced an invalid transcript.json")
                transcript_payload["inference_configuration"] = inference_configuration
                atomic_json(transcript_path, transcript_payload)
            if _source_identity(source) != guard:
                raise RuntimeError(
                    "Source media changed while transcript artifacts were being written"
                )
            state.update(
                status="completed",
                chunk_count=len(chunks),
                segment_count=len(segments),
                backend=transcript.backend,
                source_metadata=source_metadata,
                audio_metadata=audio_metadata,
                updated_at=_utc_now(),
            )
            state.pop("failed_chunks", None)
            state.pop("error", None)
            if self.config.backend == "llamacpp":
                state["inference_configuration"] = inference_configuration
            atomic_json(metadata_path, state)
            archive_path = None
            archive_warning = None
            if self.config.archive_inputs and transcript.backend != "mock":
                archive_started = time.monotonic()
                self._emit("stage.started", stage="archive")
                archive_warning = _archive_duration_warning(state)
                if archive_warning is None:
                    archive_path, archive_warning = self._archive_input(
                        source, source_folder, guard
                    )
                self._job["stages"]["archive"] = round(time.monotonic() - archive_started, 3)
                self._stage_finished("archive", self._job["stages"]["archive"])
                if archive_warning:
                    self._emit(
                        "warning",
                        code="archive_not_done",
                        message="The input was not archived; see archive_warning in metadata.json",
                    )
                if archive_path is not None:
                    state["archived_source_path"] = archive_path
                    state.pop("archive_warning", None)
                elif archive_warning:
                    state["archive_warning"] = archive_warning
                atomic_json(metadata_path, state)
            self.status(self._execution_line(checkpoint, len(chunks), duration))
            self.status(f"Completed {source.name}: {len(segments)} segments")
            result = {"job_id": job_id, "source": str(source.resolve()), "status": "completed"}
            if archive_path:
                result["archived_source_path"] = archive_path
            if archive_warning:
                result["archive_warning"] = archive_warning
            return result
        except Exception as exc:
            if intermediate is not None and self.exporter is not None:
                try:
                    saved_checkpoint = read_json(work_dir / "checkpoint.json", {}) or {}
                    self._write_intermediate(
                        intermediate,
                        segments,
                        checkpoint=saved_checkpoint if isinstance(saved_checkpoint, dict) else {},
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception as intermediate_exc:
                    self.status(f"Could not update intermediate transcript: {intermediate_exc}")
            failure = {
                "job_id": job_id,
                "source_name": source.name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": _utc_now(),
            }
            atomic_json(work_dir / "failure.json", failure)
            state.update(status="failed", error=failure["error"], updated_at=failure["updated_at"])
            atomic_json(metadata_path, state)
            self.status(f"Failed {source.name}: {failure['error']}")
            return {
                "job_id": job_id,
                "source": str(source.resolve()),
                "status": "failed",
                "error": failure["error"],
            }

    def _transcribe_pending(
        self,
        chunks: list[AudioChunk],
        work_dir: Path,
        checkpoint: dict,
        checkpoint_path: Path,
        segments: list[Segment],
        intermediate: dict,
        inference_started: float,
    ) -> None:
        """Transcribe every chunk missing from the checkpoint.

        All state, checkpoint and export writes happen on the calling thread in the order
        results are consumed. With ``parallel_requests > 1`` only ``_transcribe_chunk`` runs on
        worker threads, with at most that many chunks in flight.
        """
        done = checkpoint["chunks"]
        failed = checkpoint["failed"]
        attempts_log = checkpoint.setdefault("attempts", {})
        last_export: float | None = None
        tracker = ProgressTracker({chunk.index: chunk.end - chunk.start for chunk in chunks})
        tracker.seed(done, attempts_log)
        self._tracker = tracker
        self._emit("progress", **tracker.snapshot())

        def consume(chunk: AudioChunk, result_segments: list[Segment] | None, history: list[dict]):
            nonlocal last_export
            key = str(chunk.index)
            attempts_log[key] = history
            if result_segments is not None:
                done[key] = [_segment_dict(s) for s in result_segments]
                segments.extend(result_segments)
            else:
                failed[key] = {
                    "start": chunk.start,
                    "end": chunk.end,
                    "error": _last_error(history),
                    "attempts": history,
                }
                self.status(f"{self._chunk_label(chunk)} FAILED after ladder -> continuing")
            atomic_json(checkpoint_path, checkpoint)
            self._job["stages"]["inference"] = round(time.monotonic() - inference_started, 3)
            outcome = tracker.finish_chunk(chunk.index, result_segments, history)
            calls = [e for e in history if isinstance(e, dict) and "action" not in e]
            self._emit(
                "chunk.finished",
                index=chunk.index,
                outcome=outcome,
                attempts=len(calls),
                wall_seconds=round(sum(float(e.get("wall_seconds") or 0) for e in calls), 3),
                **tracker.counters(),
            )
            self._emit("progress", **tracker.snapshot())
            interval = self.config.intermediate_interval_seconds
            now = self._clock()
            if last_export is None or interval <= 0 or now - last_export >= interval:
                self._write_intermediate(intermediate, segments, checkpoint=checkpoint)
                last_export = self._clock()

        pending: list[AudioChunk] = []
        workers = self.config.parallel_requests
        for chunk in chunks:
            key = str(chunk.index)
            if key in done:
                segments.extend(_segment_from_dict(s) for s in done[key])
                continue
            if workers <= 1:
                # A rerun gives previously failed chunks a fresh attempt.
                failed.pop(key, None)
                result_segments, history = self._transcribe_chunk(chunk, work_dir)
                consume(chunk, result_segments, history)
            else:
                pending.append(chunk)
        if workers <= 1 or not pending:
            return

        self.status(
            f"Running up to {workers} chunk requests in parallel (the server needs at least "
            f"{workers} slots; batched GPU execution may change greedy output slightly)"
        )
        queue = iter(pending)
        in_flight: dict[Future, AudioChunk] = {}
        fatal: BaseException | None = None
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chunk") as pool:
            while True:
                while fatal is None and len(in_flight) < workers:
                    chunk = next(queue, None)
                    if chunk is None:
                        break
                    failed.pop(str(chunk.index), None)
                    in_flight[pool.submit(self._transcribe_chunk, chunk, work_dir)] = chunk
                if not in_flight:
                    break
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda item: in_flight[item].index):
                    chunk = in_flight.pop(future)
                    try:
                        result_segments, history = future.result()
                    except Exception as exc:
                        # Stop submitting; completed in-flight chunks are still saved below.
                        if fatal is None:
                            fatal = exc
                        continue
                    consume(chunk, result_segments, history)
        if fatal is not None:
            raise fatal

    def _transcribe_chunk(
        self, chunk: AudioChunk, work_dir: Path
    ) -> tuple[list[Segment] | None, list[dict]]:
        """Run the per-chunk recovery ladder.

        Transient backend errors are retried unchanged. Degenerate output moves down the
        ladder: configured temperature, each fallback temperature, then one split of the
        chunk with the same temperatures per sub-chunk. Returns ``(None, history)`` when
        every rung failed; any other error propagates and fails the job.
        """
        history: list[dict] = []
        tracker = self._tracker
        in_flight = tracker.begin(chunk.index) if tracker is not None else []
        self._emit(
            "chunk.started",
            index=chunk.index,
            part=chunk.part,
            audio_seconds=round(chunk.end - chunk.start, 3),
            in_flight=in_flight,
            in_flight_count=len(in_flight),
        )
        try:
            found = self._temperature_ladder(chunk, history)
            if found is None and self.config.split_on_failure:
                found = self._split_ladder(chunk, work_dir, history)
            return found, history
        finally:
            if tracker is not None:
                tracker.end(chunk.index)

    def _temperature_ladder(self, chunk: AudioChunk, history: list[dict]) -> list[Segment] | None:
        rungs: list[tuple[str, float | None]] = [("base", None)]
        rungs.extend(("temperature", value) for value in self.config.fallback_temperatures)
        for position, (stage, temperature) in enumerate(rungs):
            if position + 1 < len(rungs):
                after = f"fallback t={rungs[position + 1][1]:g}"
            elif self.config.context_free_fallback and self.config.prompt:
                after = "no-context retry if the context was echoed, else split"
            elif self.config.split_on_failure and not chunk.part:
                after = "split"
            else:
                after = "FAILED after ladder"
            found = self._attempt(chunk, temperature, stage, history, after)
            if found is not None:
                return found
        if (
            self.config.context_free_fallback
            and self.config.prompt
            and any(
                entry.get("outcome") == "DegenerateOutputError"
                and entry.get("part") == chunk.part
                and _degenerate_reason(entry) == "prompt_echo"
                for entry in history
            )
        ):
            # The model reproduced the context instead of speech (seen on sub-second
            # noise): one last attempt without it, recorded as its own ladder stage.
            if self.config.split_on_failure and not chunk.part:
                after = "split"
            else:
                after = "FAILED after ladder"
            found = self._attempt(chunk, None, "no_context", history, after, use_prompt=False)
            if found is not None:
                return found
        return None

    def _split_ladder(
        self, chunk: AudioChunk, work_dir: Path, history: list[dict]
    ) -> list[Segment] | None:
        try:
            parts = list(self.processor.split(chunk, work_dir / "splits"))
            _validate_split(chunk, parts, work_dir)
        except TranscriptionError as exc:
            history.append(
                {
                    "action": "split",
                    "part": chunk.part,
                    "parts": [],
                    "outcome": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return None
        history.append(
            {
                "action": "split",
                "part": chunk.part,
                "parts": [p.part for p in parts],
                "outcome": "ok",
            }
        )
        self.status(f"{self._chunk_label(chunk)} split {'/'.join(p.part for p in parts)}")
        combined: list[Segment] = []
        for part in parts:
            found = self._temperature_ladder(part, history)
            if found is None:
                return None
            combined.extend(found)
        return combined

    def _attempt(
        self,
        chunk: AudioChunk,
        temperature: float | None,
        stage: str,
        history: list[dict],
        after: str = "FAILED after ladder",
        *,
        use_prompt: bool = True,
    ) -> list[Segment] | None:
        """One ladder rung; identical retries happen only for transient errors."""
        retries = self.config.retries
        for attempt in range(retries + 1):
            started = time.monotonic()
            entry: dict = {
                "part": chunk.part,
                "stage": stage,
                "temperature": self.config.temperature if temperature is None else temperature,
                "audio_seconds": round(chunk.end - chunk.start, 3),
            }
            if not use_prompt:
                entry["context"] = "omitted"
            try:
                if use_prompt:
                    found = self.backend.transcribe(
                        chunk, language=self.config.language, temperature=temperature
                    )
                else:
                    found = self.backend.transcribe(
                        chunk,
                        language=self.config.language,
                        temperature=temperature,
                        use_prompt=False,
                    )
            except TransientBackendError as exc:
                self._finish_entry(entry, started, type(exc).__name__, exc)
                history.append(entry)
                self._emit_attempt(chunk, entry)
                if attempt < retries:
                    self.status(
                        self._attempt_line(chunk, entry, f"-> retry {attempt + 1}/{retries}")
                    )
                    continue
                self.status(self._attempt_line(chunk, entry, "-> job fails"))
                raise RuntimeError(
                    f"Transcription backend unavailable for chunk {chunk.index} after "
                    f"{retries + 1} attempt(s): {exc}"
                ) from exc
            except DegenerateOutputError as exc:
                self._finish_entry(entry, started, type(exc).__name__, exc)
                history.append(entry)
                self._emit_attempt(chunk, entry)
                self.status(self._attempt_line(chunk, entry, f"-> {after}"))
                return None
            except Exception as exc:
                self._finish_entry(entry, started, type(exc).__name__, exc)
                history.append(entry)
                self._emit_attempt(chunk, entry)
                self.status(self._attempt_line(chunk, entry, "-> job fails"))
                raise RuntimeError(f"Transcription failed for chunk {chunk.index}: {exc}") from exc
            self._finish_entry(entry, started, "ok" if found else "no_speech", None)
            history.append(entry)
            self._emit_attempt(chunk, entry)
            self.status(self._attempt_line(chunk, entry))
            return list(found)
        return None

    def _emit_attempt(self, chunk: AudioChunk, entry: dict) -> None:
        """Numbers and labels only; the model output never reaches an event."""
        metrics = entry.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        outcome = entry["outcome"]
        self._emit(
            "chunk.attempt",
            index=chunk.index,
            part=chunk.part,
            stage=entry["stage"],
            temperature=entry["temperature"],
            outcome=outcome,
            reason=_degenerate_reason(entry) if outcome == "DegenerateOutputError" else None,
            wall_seconds=entry["wall_seconds"],
            completion_tokens=metrics.get("completion_tokens"),
            finish_reason=metrics.get("finish_reason"),
            mean_token_prob=metrics.get("mean_token_prob"),
            context_omitted=entry.get("context") == "omitted",
        )

    def _execution_line(self, checkpoint: dict, total_chunks: int, duration: float) -> str:
        return _execution_line(
            checkpoint,
            total_chunks,
            duration,
            self._current_stages(),
            self.config.parallel_requests,
        )

    def _chunk_label(self, chunk: AudioChunk) -> str:
        return (
            f"Chunk {chunk.index + 1}{chunk.part}/{self._total_chunks} "
            f"[{chunk.start:.1f}-{chunk.end:.1f}s, {chunk.end - chunk.start:.1f}s audio]"
        )

    def _attempt_line(self, chunk: AudioChunk, entry: dict, after: str = "") -> str:
        """One status line per attempt; numbers and error classes only, no text."""
        outcome = entry["outcome"]
        if outcome == "DegenerateOutputError":
            outcome = f"{outcome}({_degenerate_reason(entry)})"
        metrics = entry.get("metrics")
        tokens = metrics.get("completion_tokens") if isinstance(metrics, dict) else None
        parts = [
            self._chunk_label(chunk),
            f"t={entry['temperature']:g}"
            + (" no-context" if entry.get("context") == "omitted" else ""),
            outcome,
            f"{entry['wall_seconds']:.1f}s",
        ]
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
            parts.append(f"{tokens:g} tok")
        if after:
            parts.append(after)
        return " ".join(parts)

    def _finish_entry(
        self, entry: dict, started: float, outcome: str, error: Exception | None
    ) -> None:
        entry["wall_seconds"] = round(time.monotonic() - started, 3)
        entry["outcome"] = outcome
        entry["metrics"] = self._backend_metrics()
        if error is not None:
            entry["error"] = str(error)

    def _backend_metrics(self) -> dict | None:
        reader = getattr(self.backend, "last_call_metrics", None)
        if not callable(reader):
            return None
        try:
            metrics = reader()
        except Exception:
            return None
        return dict(metrics) if isinstance(metrics, dict) else None

    def _write_intermediate(
        self,
        details: dict,
        segments: list[Segment],
        *,
        checkpoint: dict,
        status: str = "processing",
        error: str | None = None,
    ) -> None:
        if self.exporter is None:
            return
        done = checkpoint.get("chunks")
        failed = checkpoint.get("failed")
        attempts = checkpoint.get("attempts")
        done = done if isinstance(done, dict) else {}
        failed = failed if isinstance(failed, dict) else {}
        attempts = attempts if isinstance(attempts, dict) else {}
        warnings = list(details.get("warnings", []))
        if status != "completed":
            warnings.append("Partial intermediate transcript; job is not complete.")
            for index in sorted(failed, key=int):
                item = failed[index]
                warnings.append(
                    f"Gap: chunk {index} ({item['start']:.2f}-{item['end']:.2f} s) could not "
                    "be transcribed; its speech is missing from this transcript."
                )
        transcript = Transcript(
            job_id=details["job_id"],
            backend=details["backend"],
            model=details["model"],
            language=self.config.language,
            duration=details["duration"],
            segments=sorted(segments, key=lambda segment: (segment.start, segment.end)),
            warnings=warnings,
            source_metadata=details["source_metadata"],
            audio_metadata=details["audio_metadata"],
        )
        destination = self.config.output_dir / details["job_id"] / "intermediate"
        self.exporter.export(transcript, destination)
        report_path = destination / "report.json"
        report = read_json(report_path, {})
        if not isinstance(report, dict):
            raise RuntimeError("Transcript exporter produced an invalid intermediate report")
        report.update(
            status=status,
            completed_chunks=len(done),
            total_chunks=details["total_chunks"],
            failed_chunks=[
                {
                    "index": int(index),
                    "start": failed[index]["start"],
                    "end": failed[index]["end"],
                    "error": failed[index]["error"],
                }
                for index in sorted(failed, key=int)
            ],
            chunk_attempts={
                index: history
                for index, history in attempts.items()
                if isinstance(history, list)
                and (len(history) > 1 or any(item.get("outcome") != "ok" for item in history))
            },
            execution=_execution_summary(
                checkpoint,
                details["total_chunks"],
                details["duration"],
                self._current_stages(),
                self.config.parallel_requests,
            ),
            inference_configuration=self._inference_configuration(),
        )
        if error:
            report["error"] = error
        else:
            report.pop("error", None)
        atomic_json(report_path, report)
        transcript_path = destination / "transcript.json"
        transcript_payload = read_json(transcript_path, {})
        if not isinstance(transcript_payload, dict):
            raise RuntimeError("Transcript exporter produced an invalid intermediate transcript")
        transcript_payload["inference_configuration"] = self._inference_configuration()
        atomic_json(transcript_path, transcript_payload)

    def _inference_configuration(self) -> dict:
        effective_prompt = self.config.prompt
        if self.config.backend == "llamacpp" and self.backend is not None:
            resolver = getattr(self.backend, "effective_prompt", None)
            if callable(resolver):
                effective_prompt = resolver(self.config.language)
        return {
            "backend": self.config.backend,
            "model": self.config.model,
            "base_url": self.config.base_url if self.config.backend == "llamacpp" else None,
            "language": self.config.language,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "response_mode": self.config.response_mode,
            "prompt": effective_prompt if self.config.backend == "llamacpp" else None,
            "requested_prompt": self.config.prompt if self.config.backend == "llamacpp" else None,
            "max_tokens": self.config.max_tokens,
            "min_tokens": self.config.min_tokens,
            "tokens_per_second": self.config.tokens_per_second,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "repeat_penalty": self.config.repeat_penalty,
            "dry_multiplier": self.config.dry_multiplier,
            "force_language": self.config.force_language,
            "fallback_temperatures": list(self.config.fallback_temperatures),
            "split_on_failure": self.config.split_on_failure,
            "context_free_fallback": self.config.context_free_fallback,
            # Values above 1 are part of job identity: batched GPU execution may change
            # greedy output slightly.
            "parallel_requests": self.config.parallel_requests,
        }


def _archive_duration_warning(metadata: dict) -> str | None:
    source_metadata = metadata.get("source_metadata")
    audio_metadata = metadata.get("audio_metadata")
    source_duration = (
        source_metadata.get("duration_seconds") if isinstance(source_metadata, dict) else None
    )
    audio_duration = (
        audio_metadata.get("duration_seconds") if isinstance(audio_metadata, dict) else None
    )
    if not isinstance(source_duration, (int, float)) or not isinstance(
        audio_duration, (int, float)
    ):
        return (
            "Source duration is unavailable; input was retained to avoid archiving partial audio."
        )
    if not math.isfinite(source_duration) or not math.isfinite(audio_duration):
        return "Source duration is invalid; input was retained to avoid archiving partial audio."
    sample_rate = audio_metadata.get("sample_rate", 16000)
    frame = 1 / sample_rate if isinstance(sample_rate, int) and sample_rate > 0 else 0
    tolerance = max(frame, ARCHIVE_DURATION_TOLERANCE)
    if audio_duration + tolerance < source_duration:
        return "Only part of the source was processed; input remains queued."
    return None


class ProcessLock:
    """OS-backed nonblocking lock; a crashed process releases it automatically."""

    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError("Another audio-transcript process is already running") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()


def run_watch(
    pipeline: TranscriptionPipeline,
    *,
    interval: float,
    stable_scans: int,
    stop: Callable[[], bool] | None = None,
) -> None:
    """Poll for new or changed files and process only after stable observations."""
    previous: dict[Path, tuple[int, int, int]] = {}
    stable: dict[Path, int] = {}
    completed: dict[Path, tuple[int, int]] = {}
    attempts: dict[Path, tuple[tuple[int, int], int]] = {}
    stop = stop or (lambda: False)
    while not stop():
        current = {}
        ready = []
        ready_signatures: dict[Path, tuple[int, int]] = {}
        for source in pipeline.discover():
            try:
                stat = source.stat()
            except OSError:
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            current[source] = (*signature, stable.get(source, 0))
            old = previous.get(source)
            count = stable.get(source, 0) + 1 if old and old[:2] == signature else 1
            stable[source] = count
            if attempts.get(source, (signature, 0))[0] != signature:
                attempts.pop(source, None)
            attempt_count = attempts.get(source, (signature, 0))[1]
            max_attempts = getattr(getattr(pipeline, "config", None), "retries", 2) + 1
            if (
                count >= stable_scans
                and completed.get(source) != signature
                and attempt_count < max_attempts
            ):
                ready.append(source)
                ready_signatures[source] = signature
        previous = {p: (v[0], v[1], v[2]) for p, v in current.items()}
        if ready:
            results = pipeline.run(ready, close_backend=False)
            if not isinstance(results, list):
                results = [{"status": "failed"} for _ in ready]
            for source, result in zip(ready, results, strict=False):
                signature = ready_signatures[source]
                status = result.get("status") if isinstance(result, dict) else None
                if status in {"completed", "skipped", "prepared"}:
                    completed[source] = signature
                    attempts.pop(source, None)
                else:
                    old_signature, count = attempts.get(source, (signature, 0))
                    count = count + 1 if old_signature == signature else 1
                    attempts[source] = (signature, count)
                    max_attempts = getattr(getattr(pipeline, "config", None), "retries", 2) + 1
                    if count >= max_attempts:
                        completed[source] = signature
        time.sleep(interval)


def _status_level(message: str) -> str:
    if message.startswith("Failed ") or "FAILED" in message or "-> job fails" in message:
        return "error"
    if message.startswith(("Could not", "Incomplete")) or (
        message.startswith("Skipped") and "already completed" not in message
    ):
        return "warning"
    return "info"


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _scope(max_duration: float | None) -> dict:
    return {
        "kind": "full" if max_duration is None else "bounded",
        "max_duration_seconds": max_duration,
        "label": scope_label(max_duration),
    }


def _atomic_text(path: Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_identity(source: Path) -> dict:
    stat = source.stat()
    return {"source": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _path_signature(path: Path | None) -> dict | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "size": None, "mtime_ns": None}
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _same_file_bytes(first: Path, second: Path) -> bool:
    try:
        if first.stat().st_size != second.stat().st_size:
            return False
        digests = []
        for path in (first, second):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            digests.append(digest.digest())
        return digests[0] == digests[1]
    except OSError:
        return False


def _valid_checkpoint(value: object, fingerprint: str, available_chunks: list[AudioChunk]) -> bool:
    if not isinstance(value, dict) or value.get("fingerprint") != fingerprint:
        return False
    chunks = value.get("chunks")
    if not isinstance(chunks, dict):
        return False
    chunks_by_index = {str(chunk.index): chunk for chunk in available_chunks}
    if set(chunks) - chunks_by_index.keys():
        return False
    failed = value.get("failed", {})
    attempts = value.get("attempts", {})
    if not isinstance(failed, dict) or not isinstance(attempts, dict):
        return False
    if set(failed) - chunks_by_index.keys() or set(failed) & set(chunks):
        return False
    if set(attempts) - chunks_by_index.keys():
        return False
    for item in failed.values():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("error"), str)
            or not isinstance(item.get("attempts"), list)
        ):
            return False
        try:
            if not math.isfinite(float(item["start"])) or not math.isfinite(float(item["end"])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    if not all(
        isinstance(history, list) and all(isinstance(entry, dict) for entry in history)
        for history in attempts.values()
    ):
        return False
    try:
        for index, segments in chunks.items():
            if not isinstance(segments, list):
                return False
            chunk = chunks_by_index[index]
            previous_end = chunk.start
            for segment in segments:
                if not isinstance(segment, dict):
                    return False
                parsed = _segment_from_dict(segment)
                if (
                    not math.isfinite(parsed.start)
                    or not math.isfinite(parsed.end)
                    or parsed.start < chunk.start - 0.000001
                    or parsed.end > chunk.end + 0.000001
                    or parsed.start < previous_end - 0.000001
                    or parsed.end <= parsed.start
                    or not parsed.text.strip()
                    or (parsed.speaker is not None and not isinstance(parsed.speaker, str))
                    or not isinstance(parsed.timing_source, str)
                ):
                    return False
                previous_end = parsed.end
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _validate_prepared(
    prepared: PreparedAudio, work_dir: Path, max_duration: float | None
) -> float:
    """Validate preparation output and return the effective analysed duration."""
    chunks = list(prepared.chunks)
    duration = prepared.duration
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Media preparation returned an invalid audio duration")
    if not math.isfinite(prepared.speech_seconds) or prepared.speech_seconds < 0:
        raise RuntimeError("Media preparation returned an invalid speech duration")
    if not isinstance(prepared.segmentation, dict):
        raise RuntimeError("Media preparation returned invalid segmentation provenance")
    previous_end = 0.0
    for expected_index, chunk in enumerate(chunks):
        path = Path(chunk.path).resolve()
        try:
            path.relative_to(work_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "Prepared audio chunk was written outside its job directory"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"Prepared audio chunk is missing: {path.name}")
        if (
            not math.isfinite(chunk.start)
            or not math.isfinite(chunk.end)
            or chunk.start < previous_end - 0.000001
            or chunk.start < 0
            or chunk.end <= chunk.start
            or chunk.index != expected_index
            or chunk.part != ""
        ):
            raise RuntimeError("Media preparation returned invalid or unordered audio chunks")
        if max_duration is not None and chunk.end > max_duration + 0.0001:
            raise RuntimeError("Media preparation exceeded max_duration")
        previous_end = chunk.end
    if chunks and chunks[-1].end > duration + 0.05:
        raise RuntimeError("Prepared audio chunks extend beyond the reported audio duration")
    # Rounding can leave the last chunk marginally past the reported duration.
    return max(duration, chunks[-1].end) if chunks else duration


def _validate_split(parent: AudioChunk, parts: list[AudioChunk], work_dir: Path) -> None:
    """Reject sub-chunks that do not cover the parent span in order."""
    if len(parts) < 2:
        raise TranscriptionError("Chunk split returned fewer than two sub-chunks")
    labels = [part.part for part in parts]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise TranscriptionError("Chunk split returned sub-chunks without distinct part labels")
    previous_end = parent.start
    for part in parts:
        try:
            Path(part.path).resolve().relative_to(work_dir.resolve())
            exists = Path(part.path).is_file()
        except (ValueError, OSError):
            exists = False
        if (
            not exists
            or part.index != parent.index
            or not math.isfinite(part.start)
            or not math.isfinite(part.end)
            or part.end <= part.start
            or part.start < previous_end - 0.001
            or part.start < parent.start - 0.001
            or part.end > parent.end + 0.001
        ):
            raise TranscriptionError("Chunk split returned invalid or unordered sub-chunks")
        previous_end = part.end


def _degenerate_reason(entry: dict) -> str:
    """Short, text-free reason for a degenerate attempt."""
    metrics = entry.get("metrics")
    if isinstance(metrics, dict) and metrics.get("finish_reason") == "length":
        return "length"
    message = str(entry.get("error", ""))
    if "repetition loop" in message:
        return "loop"
    if "instruction prompt" in message:
        return "prompt_echo"
    return "invalid"


def _prepared_summary(
    name: str, prepared: PreparedAudio, duration: float, prep_seconds: float | None
) -> str:
    lengths = [chunk.end - chunk.start for chunk in prepared.chunks]
    detector = prepared.segmentation.get("detector", "unknown")
    share = 100 * prepared.speech_seconds / duration if duration > 0 else 0.0
    if lengths:
        spread = (
            f"{len(lengths)} chunks, length min/avg/max "
            f"{min(lengths):.1f}/{sum(lengths) / len(lengths):.1f}/{max(lengths):.1f}s"
        )
    else:
        spread = "0 chunks"
    took = f"{prep_seconds:.1f}s" if prep_seconds is not None else "reused from earlier run"
    return (
        f"Prepared {name}: {duration:.1f}s analysed, detector={detector}, "
        f"speech {prepared.speech_seconds:.1f}s ({share:.0f}%), {spread}, preparation {took}"
    )


def _execution_line(
    checkpoint: dict,
    total_chunks: int,
    duration: float,
    stages: dict | None = None,
    parallel_requests: int | None = None,
) -> str:
    execution = _execution_summary(checkpoint, total_chunks, duration, stages, parallel_requests)
    tokens = execution["completion_tokens"]
    rtf = execution["real_time_factor"]
    maximum = tokens["max"] if tokens["max"] is not None else "n/a"
    p95 = tokens["p95"] if tokens["p95"] is not None else "n/a"
    extra = ""
    if execution["inference_stage_wall_seconds"] is not None:
        extra = f", stage wall {execution['inference_stage_wall_seconds']:.1f}s"
        if execution["effective_concurrency"] is not None:
            extra += f", concurrency {execution['effective_concurrency']:g}x"
    return (
        f"Execution: chunks ok={execution['chunks_ok']} "
        f"no_speech={execution['chunks_no_speech']} failed={execution['chunks_failed']}, "
        f"inference {execution['inference_wall_seconds']:.1f}s "
        f"(RTF {rtf if rtf is not None else 'n/a'}){extra}, tokens sum={tokens['sum']:g} "
        f"max={maximum} p95={p95}, "
        f"fallbacks={execution['temperature_fallbacks_used']} splits={execution['splits_used']}"
    )


def _last_error(history: list[dict]) -> str:
    for entry in reversed(history):
        if entry.get("error"):
            return f"{entry.get('outcome', 'error')}: {entry['error']}"
    return "no usable transcription"


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


LOW_CONFIDENCE_THRESHOLD = 0.5


def _chunk_confidences(done: dict, attempts: dict) -> list[tuple[float, float]]:
    """Per completed chunk: (mean token probability, token weight) from its ok attempts."""
    found = []
    for key in done:
        history = attempts.get(key)
        if not isinstance(history, list):
            continue
        total = weight = 0.0
        for entry in history:
            metrics = entry.get("metrics") if isinstance(entry, dict) else None
            if not isinstance(metrics, dict) or entry.get("outcome") != "ok":
                continue
            mean, count = metrics.get("mean_token_prob"), metrics.get("logprob_tokens")
            if isinstance(mean, bool) or not isinstance(mean, (int, float)):
                continue
            count = count if isinstance(count, (int, float)) and count > 0 else 1
            total += mean * count
            weight += count
        if weight:
            found.append((total / weight, weight))
    return found


def _confidence_summary(done: dict, attempts: dict) -> dict | None:
    """Uncalibrated proxy: token-weighted mean probability of the generated tokens."""
    chunks = _chunk_confidences(done, attempts)
    if not chunks:
        return None
    means = [mean for mean, _ in chunks]
    weight = sum(w for _, w in chunks)
    return {
        "method": "mean_token_probability",
        "calibrated": False,
        "mean": round(sum(m * w for m, w in chunks) / weight, 4),
        "p10_chunk": round(_percentile(means, 0.1), 4),
        "low_chunks": sum(1 for mean in means if mean < LOW_CONFIDENCE_THRESHOLD),
    }


def _execution_summary(
    checkpoint: dict,
    total_chunks: int,
    analysed_seconds: float,
    stages: dict | None = None,
    parallel_requests: int | None = None,
) -> dict:
    """Aggregate attempt history; contains numbers only, never transcript or prompt text."""
    done = checkpoint.get("chunks")
    failed = checkpoint.get("failed")
    attempts = checkpoint.get("attempts")
    done = done if isinstance(done, dict) else {}
    failed = failed if isinstance(failed, dict) else {}
    attempts = attempts if isinstance(attempts, dict) else {}
    entries = [
        entry
        for history in attempts.values()
        if isinstance(history, list)
        for entry in history
        if isinstance(entry, dict)
    ]
    calls = [entry for entry in entries if "action" not in entry]
    wall = sum(float(e.get("wall_seconds") or 0) for e in calls)
    audio = sum(float(e.get("audio_seconds") or 0) for e in calls)
    tokens = []
    for entry in calls:
        metrics = entry.get("metrics")
        value = metrics.get("completion_tokens") if isinstance(metrics, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tokens.append(value)
    chunk_seconds = [
        sum(
            float(e.get("wall_seconds") or 0)
            for e in history
            if isinstance(e, dict) and "action" not in e
        )
        for history in attempts.values()
        if isinstance(history, list)
    ]
    stage_wall = stages.get("inference") if isinstance(stages, dict) else None
    if isinstance(stage_wall, bool) or not isinstance(stage_wall, (int, float)):
        stage_wall = None
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "parallel_requests": parallel_requests,
        "total_chunks": total_chunks,
        "chunks_ok": sum(1 for segments in done.values() if segments),
        "chunks_no_speech": sum(1 for segments in done.values() if not segments),
        "chunks_failed": len(failed),
        "audio_seconds_sent": round(audio, 3),
        # Sum of the model request wall times; exceeds stage wall time when requests overlap.
        "inference_wall_seconds": round(wall, 3),
        "request_seconds_sum": round(wall, 3),
        "inference_stage_wall_seconds": stage_wall,
        "effective_concurrency": (
            round(wall / stage_wall, 2) if stage_wall is not None and stage_wall > 0 else None
        ),
        "analysed_audio_seconds": analysed_seconds,
        "real_time_factor": round(wall / analysed_seconds, 4) if analysed_seconds > 0 else None,
        "completion_tokens": {
            "calls_with_metrics": len(tokens),
            "sum": sum(tokens),
            "max": max(tokens) if tokens else None,
            "p95": _percentile(tokens, 0.95),
        },
        "chunk_inference_seconds": {
            "mean": round(sum(chunk_seconds) / len(chunk_seconds), 3) if chunk_seconds else None,
            "p95": _percentile([round(v, 3) for v in chunk_seconds], 0.95),
            "max": round(max(chunk_seconds), 3) if chunk_seconds else None,
        },
        "confidence": _confidence_summary(done, attempts),
        "stages_seconds": dict(stages) if stages else None,
        "attempts": len(calls),
        "temperature_fallbacks_used": sum(1 for e in calls if e.get("stage") == "temperature"),
        "splits_used": sum(
            1 for e in entries if e.get("action") == "split" and e.get("outcome") == "ok"
        ),
    }
