"""Sequential, resumable media ingestion and transcription pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..config import AppConfig
from ..domain.models import AudioChunk, Segment, Transcript
from ..domain.ports import MediaProcessor, TranscriptExporter, TranscriptionBackend
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
    ):
        self.config = config
        self.processor = processor
        self.backend = backend
        self.exporter = exporter
        self.status = status

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
        stat = source.stat()
        identity = {
            "source": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        digest = hashlib.sha256(
            json.dumps(
                {"source": identity, "config_fingerprint": fingerprint}, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:14]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._-") or "media"
        return f"{stem[:64]}-{digest}", identity

    def _fingerprint(self) -> str:
        data = {
            "backend": self.config.backend,
            "model": self.config.model,
            "model_path": _path_signature(self.config.model_path),
            "projector_path": _path_signature(self.config.projector_path),
            "device": self.config.device,
            "language": self.config.language,
            "chunk_seconds": self.config.chunk_seconds,
            "sample_rate": self.config.sample_rate,
            "max_duration": self.config.max_duration,
            "backend_settings": (
                {
                    "base_url": self.config.base_url,
                    "timeout": self.config.timeout,
                    "max_tokens": self.config.max_tokens,
                    "prompt": self.config.prompt,
                    "temperature": self.config.temperature,
                    "seed": self.config.seed,
                    "response_mode": self.config.response_mode,
                }
                if self.config.backend == "llamacpp"
                else None
            ),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_chunks(self, job_dir: Path, fingerprint: str) -> list[AudioChunk] | None:
        manifest = read_json(job_dir / "prepared.json")
        if not isinstance(manifest, dict) or manifest.get("fingerprint") != fingerprint:
            return None
        try:
            items = manifest["chunks"]
            if not isinstance(items, list) or not items:
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
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or start < previous_end - 0.000001
                    or end <= start
                    or index != expected_index
                ):
                    return None
                chunks.append(AudioChunk(path, start, end, index))
                previous_end = end
            return chunks
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
                result.setdefault("source", str(source.resolve()))
                results[index] = result
        finally:
            if close_backend and self.backend is not None:
                self.backend.close()
        return [result for result in results if result is not None]

    def _record_rejected(self, source: Path, fingerprint: str, message: str) -> None:
        try:
            job_id, identity = self._identity(source, fingerprint)
        except OSError:
            return
        now = _utc_now()
        failure = {
            "job_id": job_id,
            "source_name": source.name,
            "status": "failed",
            "error": message,
            "updated_at": now,
        }
        atomic_json(self.config.process_dir / job_id / "failure.json", failure)
        atomic_json(
            self.config.process_dir / job_id / "metadata.json",
            {
                "job_id": job_id,
                "identity": identity,
                "config_fingerprint": fingerprint,
                "source_name": source.name,
                "status": "failed",
                "error": message,
                "updated_at": now,
            },
        )

    def _archive_input(
        self, source: Path, job_id: str, identity: dict
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
            if _source_identity(source) != identity:
                raise RuntimeError("Source changed after transcription; archive was skipped")
            archive_dir = self.config.processed_dir / job_id
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
                if _source_identity(source) != identity:
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
        job_id, identity = self._identity(source, fingerprint)
        work_dir = self.config.process_dir / job_id
        output_dir = self.config.output_dir / job_id
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
            and metadata.get("identity") == identity
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
                    archive_path, warning = self._archive_input(source, job_id, identity)
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

        state = {
            "job_id": job_id,
            "identity": identity,
            "config_fingerprint": fingerprint,
            "source_name": source.name,
            "status": "processing",
            "updated_at": _utc_now(),
        }
        if self.config.backend == "llamacpp":
            state["inference_configuration"] = self._inference_configuration()
        atomic_json(metadata_path, state)
        work_dir.mkdir(parents=True, exist_ok=True)
        segments: list[Segment] = []
        intermediate: dict | None = None
        try:
            info = self.processor.probe(source)
            chunks = None if self.config.force else self._load_chunks(work_dir, fingerprint)
            rebuilt_chunks = chunks is None
            if chunks is None:
                prepared = self.processor.prepare(
                    source,
                    work_dir,
                    chunk_seconds=self.config.chunk_seconds,
                    sample_rate=self.config.sample_rate,
                    max_duration=self.config.max_duration,
                )
                chunks = list(prepared)
                if not chunks:
                    raise RuntimeError("Media preparation returned no audio chunks")
                _validate_chunks(chunks, work_dir, self.config.max_duration)
                if _source_identity(source) != identity:
                    raise RuntimeError("Source media changed while audio preparation was running")
                atomic_json(
                    work_dir / "prepared.json",
                    {
                        "fingerprint": fingerprint,
                        "source_duration": info.duration,
                        "duration": chunks[-1].end,
                        "chunks": [
                            {
                                "path": str(c.path.resolve()),
                                "start": c.start,
                                "end": c.end,
                                "index": c.index,
                            }
                            for c in chunks
                        ],
                    },
                )
            audio_metadata = {
                "sample_rate": self.config.sample_rate,
                "channels": 1,
                "codec": "pcm_s16le",
                "duration_seconds": chunks[-1].end,
                "chunk_count": len(chunks),
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
            if self.config.prepare_only:
                self.status(f"Prepared {source.name} ({len(chunks)} chunks)")
                return {"job_id": job_id, "source": str(source.resolve()), "status": "prepared"}
            if self.backend is None or self.exporter is None:
                raise RuntimeError("Backend or transcript exporter is unavailable")

            intermediate = {
                "job_id": job_id,
                "backend": getattr(self.backend, "name", self.config.backend),
                "model": self.config.model,
                "duration": chunks[-1].end,
                "source_metadata": source_metadata,
                "audio_metadata": audio_metadata,
                "total_chunks": len(chunks),
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
            for chunk in chunks:
                key = str(chunk.index)
                if key in done:
                    segments.extend(_segment_from_dict(s) for s in done[key])
                    self._write_intermediate(intermediate, segments, completed_chunks=len(done))
                    self.status(
                        f"Chunk {chunk.index + 1}/{len(chunks)} completed for {source.name}"
                    )
                    continue
                error = None
                for attempt in range(self.config.retries + 1):
                    try:
                        current = self.backend.transcribe(chunk, language=self.config.language)
                        done[key] = [_segment_dict(s) for s in current]
                        atomic_json(checkpoint_path, checkpoint)
                        segments.extend(current)
                        self._write_intermediate(intermediate, segments, completed_chunks=len(done))
                        self.status(
                            f"Chunk {chunk.index + 1}/{len(chunks)} completed for {source.name}"
                        )
                        error = None
                        break
                    except Exception as exc:
                        error = exc
                        if attempt < self.config.retries:
                            self.status(
                                f"Retry {attempt + 1}/{self.config.retries} for {source.name} chunk {chunk.index}"
                            )
                if error is not None:
                    raise RuntimeError(
                        f"Transcription failed for chunk {chunk.index}: {error}"
                    ) from error

            duration = chunks[-1].end
            transcript = Transcript(
                job_id=job_id,
                backend=getattr(self.backend, "name", self.config.backend),
                model=self.config.model,
                language=self.config.language,
                duration=duration,
                segments=sorted(segments, key=lambda segment: (segment.start, segment.end)),
                source_metadata=source_metadata,
                audio_metadata=audio_metadata,
            )
            if _source_identity(source) != identity:
                raise RuntimeError("Source media changed while transcription was running")
            self.exporter.export(transcript, output_dir)
            if intermediate is not None:
                self._write_intermediate(
                    intermediate, segments, completed_chunks=len(chunks), status="completed"
                )
            inference_configuration = self._inference_configuration()
            if self.config.backend == "llamacpp":
                report_path = output_dir / "report.json"
                report = read_json(report_path, {})
                if not isinstance(report, dict):
                    raise RuntimeError("Transcript exporter produced an invalid report.json")
                report["inference_configuration"] = inference_configuration
                atomic_json(report_path, report)
                transcript_path = output_dir / "transcript.json"
                transcript_payload = read_json(transcript_path, {})
                if not isinstance(transcript_payload, dict):
                    raise RuntimeError("Transcript exporter produced an invalid transcript.json")
                transcript_payload["inference_configuration"] = inference_configuration
                atomic_json(transcript_path, transcript_payload)
            if _source_identity(source) != identity:
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
            if self.config.backend == "llamacpp":
                state["inference_configuration"] = inference_configuration
            atomic_json(metadata_path, state)
            archive_path = None
            archive_warning = None
            if self.config.archive_inputs and transcript.backend != "mock":
                archive_warning = _archive_duration_warning(state)
                if archive_warning is None:
                    archive_path, archive_warning = self._archive_input(source, job_id, identity)
                if archive_path is not None:
                    state["archived_source_path"] = archive_path
                    state.pop("archive_warning", None)
                elif archive_warning:
                    state["archive_warning"] = archive_warning
                atomic_json(metadata_path, state)
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
                    saved_chunks = (
                        saved_checkpoint.get("chunks", {})
                        if isinstance(saved_checkpoint, dict)
                        else {}
                    )
                    self._write_intermediate(
                        intermediate,
                        segments,
                        completed_chunks=(
                            len(saved_chunks) if isinstance(saved_chunks, dict) else 0
                        ),
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

    def _write_intermediate(
        self,
        details: dict,
        segments: list[Segment],
        *,
        completed_chunks: int,
        status: str = "processing",
        error: str | None = None,
    ) -> None:
        if self.exporter is None:
            return
        transcript = Transcript(
            job_id=details["job_id"],
            backend=details["backend"],
            model=details["model"],
            language=self.config.language,
            duration=details["duration"],
            segments=sorted(segments, key=lambda segment: (segment.start, segment.end)),
            warnings=(
                []
                if status == "completed"
                else ["Partial intermediate transcript; job is not complete."]
            ),
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
            completed_chunks=completed_chunks,
            total_chunks=details["total_chunks"],
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
    tolerance = 1 / sample_rate if isinstance(sample_rate, int) and sample_rate > 0 else 0
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


def _validate_chunks(chunks: list[AudioChunk], work_dir: Path, max_duration: float | None) -> None:
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
        ):
            raise RuntimeError("Media preparation returned invalid or unordered audio chunks")
        if max_duration is not None and chunk.end > max_duration + 0.0001:
            raise RuntimeError("Media preparation exceeded max_duration")
        previous_end = chunk.end
