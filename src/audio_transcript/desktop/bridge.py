"""Narrow desktop bridge. File paths come from the queue or the native picker."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from collections import OrderedDict, deque
from dataclasses import replace
from pathlib import Path

from ..adapters.desktop_media import (
    WaveformCancelled,
    hidden_process,
    import_media,
    open_folder,
    waveform,
)
from ..adapters.ffmpeg import FFmpegProcessor
from ..application.delivery import ArtifactDelivery
from ..application.pipeline import MEDIA_SUFFIXES, TranscriptionPipeline
from ..application.wizard import MAX_CONTEXT_CHARS, validate_language
from ..config import (
    AFTER_SUCCESS_CHOICES,
    DEFAULT_OUTPUT_FORMATS,
    load_config,
    parse_output_formats,
)
from .playback import PlaybackServer
from .results import inspect_files, queued_file_outcome

PEAK_CACHE_SIZE = 8


def selection_options(payload: dict, duration: float, response_mode: str) -> dict:
    """Validate the same request used for recap and execution; clamp only rounding."""
    language, error = validate_language(str(payload.get("language", "auto")), response_mode)
    if error:
        raise ValueError(error)
    context = payload.get("context", "")
    if not isinstance(context, str) or len(context) > MAX_CONTEXT_CHARS:
        raise ValueError(f"Context must be at most {MAX_CONTEXT_CHARS} characters")
    numbers = []
    for key in ("start", "end"):
        raw = payload.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError("Start and end must be finite seconds")
        numbers.append(float(raw))
    start, end = numbers
    if not 0 <= start < end <= duration + 0.001 or end - start < 0.05:
        raise ValueError("Choose an interval of at least 0.05 seconds within the recording")
    end = min(end, duration)
    # An exact full-range selection uses the same identity as the CLI full run.
    full = start == 0 and abs(end - duration) < 0.001
    policy = payload.get("after_success", "keep-all")
    if policy not in AFTER_SUCCESS_CHOICES:
        raise ValueError("Unknown retention policy")
    if not full and policy != "keep-all":
        raise ValueError("Selected intervals always retain the queued recording")
    output_formats = parse_output_formats(payload.get("output_formats", DEFAULT_OUTPUT_FORMATS))
    return {
        "language": language or "auto",
        "prompt": context.strip() or None,
        "start_seconds": start,
        "max_duration": None if full else end - start,
        "after_success": policy,
        "archive_inputs": True,
        "prepare_only": False,
        "force": False,
        "output_formats": output_formats,
        "archive_intermediate": True,
    }


class DesktopBridge:
    def __init__(self, config_path: Path):
        self._config_path = config_path.resolve()
        self._config = load_config(self._config_path)
        self._config = replace(
            self._config,
            **{
                key: getattr(self._config, key).resolve()
                for key in ("input_dir", "process_dir", "output_dir", "processed_dir")
            },
        )
        self._processor = FFmpegProcessor(self._config.ffmpeg, self._config.ffprobe)
        self._window = None
        self._files: dict[str, Path] = {}
        self._selected: dict | None = None
        self._operation = threading.Lock()
        self._state_lock = threading.RLock()
        self._running = False
        self._state: dict = {"running": False, "phase": "idle"}
        self._messages = deque(maxlen=12)
        self._result_folder: Path | None = None
        self._recap: tuple[dict, tuple] | None = None
        # Session map: queue copy made by import_file -> (external source, size, mtime_ns).
        self._imported: dict[Path, tuple[Path, int, int]] = {}
        self._wave_lock = threading.Lock()
        self._wave: tuple[threading.Event, threading.Event] | None = None
        self._peaks: OrderedDict[tuple, dict] = OrderedDict()
        self._review_job: str | None = None
        self._result_job: tuple[str, str | None] | None = None
        self._encoder_problem: str | None = None
        self._candidate_serial = 0
        self._playback_server: PlaybackServer | None = None
        self._candidates: list[tuple[Path, float, float | None, str, tuple | None]] = []

    def queue(self) -> dict:
        files = TranscriptionPipeline(
            self._config, self._processor, status=lambda _: None
        ).discover()
        mapping = {str(i): path.resolve() for i, path in enumerate(files)}
        with self._state_lock:
            if self._running:
                raise RuntimeError("A transcription is already running")
            self._files = mapping
        return {
            "files": [
                {"id": key, "name": path.name, "bytes": path.stat().st_size}
                for key, path in mapping.items()
            ],
            "language": self._config.language or "auto",
            "context": self._config.prompt or "",
            "model": self._config.model,
            "batch": False,
        }

    def select(self, file_id: str) -> dict:
        with self._operation:
            if self._running:
                raise RuntimeError("A transcription is already running")
            path = self._files.get(str(file_id))
            if path is None or not path.is_file():
                raise ValueError("Refresh the queue and choose a recording")
            self._stop_waveform()
            self.playback_stop()
            info = self._processor.probe(path)
            self._selected = {
                "path": path,
                "duration": info.duration,
                "identity": self._identity(path),
                "has_video": info.has_video,
            }
            self._recap = None
            return {
                "id": file_id,
                "name": path.name,
                "duration": info.duration,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                # Only a copy made in this session is preselected for deletion; a file
                # that was already in input may be the user's only copy.
                "default_after_success": "delete-all" if path in self._imported else "keep-all",
                "has_video": info.has_video,
                "mp4_encoder": not info.has_video or self._video_encoder_problem() is None,
            }

    def import_file(self) -> dict:
        with self._operation:
            if self._running or self._window is None:
                raise RuntimeError("File import is unavailable during a run")
            import webview

            files = self._window.create_file_dialog(
                webview.FileDialog.OPEN, allow_multiple=False, file_types=("Media files (*.*)",)
            )
            if not files:
                return {"cancelled": True}
            source = Path(files[0])
            if source.suffix.lower() not in MEDIA_SUFFIXES:
                raise ValueError("Choose a supported audio or video file")
            self._processor.probe(source)
            before = source.stat()
            imported = import_media(source, self._config.input_dir).resolve()
            if imported != source.resolve():  # a file picked from inside the queue is no copy
                self._imported[imported] = (source.resolve(), before.st_size, before.st_mtime_ns)
            result = self.queue()
            result["imported_id"] = next(
                key for key, path in self._files.items() if path == imported
            )
            return result

    def preview(self) -> dict:
        """Decode a waveform outside the operation lock; select, start and close cancel it."""
        while True:
            self._stop_waveform()  # only one decode runs at a time
            with self._operation:  # held only to read and register the source
                source = self._checked_source()
                if self._running:
                    raise RuntimeError("Preview is unavailable during a run")
                identity = self._selected["identity"]
                if identity in self._peaks:
                    self._peaks.move_to_end(identity)
                    return self._peaks[identity]
                cancel, done = threading.Event(), threading.Event()
                with self._wave_lock:
                    if self._wave is None:
                        self._wave = (cancel, done)
                        break
        try:
            scratch = self._config.process_dir / "desktop-preview"
            data = waveform(self._processor, source, scratch, cancel=cancel)
        except WaveformCancelled:
            return {"cancelled": True}
        finally:
            done.set()
            with self._wave_lock:
                if self._wave is not None and self._wave[1] is done:
                    self._wave = None
        self._peaks[identity] = data
        while len(self._peaks) > PEAK_CACHE_SIZE:
            self._peaks.popitem(last=False)
        return data

    def _stop_waveform(self) -> None:
        """Cancel the running decode and wait until FFmpeg has released the source."""
        with self._wave_lock:
            current = self._wave
        if current is not None:
            current[0].set()
            current[1].wait()

    def shutdown(self) -> None:
        """Release FFmpeg and the media server; safe to call repeatedly."""
        self._stop_waveform()
        if self._playback_server is not None:
            self._playback_server.close()
            self._playback_server = None

    def _video_encoder_problem(self) -> str | None:
        if self._encoder_problem is None:
            check = getattr(self._processor, "check_video_encoder", None)
            try:
                if check is not None:
                    check()
                self._encoder_problem = ""
            except Exception as exc:  # TranscriptionError carries the actionable text
                self._encoder_problem = str(exc) or "FFmpeg cannot render video"
        return self._encoder_problem or None

    def _check_mp4(self, options: dict) -> None:
        """Say so before Start, not after hours of transcription."""
        if "mp4" not in options["output_formats"]:
            return
        if not self._selected["has_video"]:
            raise ValueError("MP4 needs a recording with a video stream; deselect MP4.")
        problem = self._video_encoder_problem()
        if problem:
            raise ValueError(problem)

    @staticmethod
    def _identity(path: Path) -> tuple:
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)

    def _checked_source(self) -> Path:
        selected = self._selected
        if selected is None:
            raise ValueError("Choose a recording first")
        source = selected["path"]
        if self._identity(source) != selected["identity"]:
            raise ValueError("The recording changed; select it again")
        return source

    def review(self, payload: dict) -> dict:
        with self._operation:
            source = self._checked_source()
            options = selection_options(
                payload, self._selected["duration"], self._config.response_mode
            )
            self._check_mp4(options)
            self._recap = (dict(options), self._identity(source))
            return {
                "file": source.name,
                "model": self._config.model,
                "options": options,
                "duration": options["max_duration"] or self._selected["duration"],
                "full": options["max_duration"] is None,
            }

    def start(self, payload: dict) -> dict:
        with self._operation:
            if self._running:
                raise RuntimeError("A transcription is already running")
            source = self._checked_source()
            options = selection_options(
                payload, self._selected["duration"], self._config.response_mode
            )
            if self._recap != (options, self._identity(source)):
                raise ValueError("Review these settings before starting")
            self._check_mp4(options)
            # FFmpeg or the media server must not hold the source while the worker archives it.
            self._stop_waveform()
            self.playback_stop()
            overrides = {
                **options,
                "ui": "jsonl",
                **{
                    key: str(getattr(self._config, key))
                    for key in ("input_dir", "process_dir", "output_dir", "processed_dir")
                },
            }
            request = {
                "config": str(self._config_path),
                "source": str(source),
                "overrides": overrides,
            }
            with self._state_lock:
                self._running = True
                self._messages.clear()
                self._result_folder = None
                self._review_job = None
                self._result_job = None
                self._state = {"running": True, "phase": "starting", "progress": {}, "result": None}
            threading.Thread(target=self._work, args=(request,), daemon=False).start()
            return {"started": True}

    def _work(self, request: dict) -> None:
        process = None
        try:
            logs = self._config.process_dir / "desktop"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / "worker-stderr.log").open("w", encoding="utf-8") as errors:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "audio_transcript.desktop.worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    text=True,
                    encoding="utf-8",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                    **hidden_process(),
                )
                process.stdin.write(json.dumps(request))
                process.stdin.close()
                with process.stdout:
                    for line in process.stdout:
                        try:
                            self._consume(json.loads(line))
                        except (ValueError, TypeError, AttributeError):
                            # A stray non-event line must never stop a recoverable run.
                            continue
                code = process.wait()
                with self._state_lock:
                    if self._state.get("result") is None:
                        raise RuntimeError(
                            f"The worker exited without a result (code {code}); see process/desktop/worker-stderr.log"
                        )
        except Exception as exc:
            with self._state_lock:
                self._state["result"] = {"ok": False, "error": {"message": str(exc)}}
        finally:
            try:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                with self._state_lock:
                    self._running = False
                    self._state["running"] = False
                    self._state["phase"] = "finished"

    def _consume(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        kind, data = event.get("type"), event.get("data")
        if not isinstance(data, dict):
            data = {}
        with self._state_lock:
            if kind == "result":
                self._state["result"] = event
                jobs = event.get("jobs")
                for job in jobs if isinstance(jobs, list) else []:
                    if not isinstance(job, dict):
                        continue
                    parts = str(job.get("job_id", "")).split("/")
                    if len(parts) == 2 and all(p not in {"", ".", ".."} for p in parts):
                        root = self._config.output_dir.resolve()
                        folder = (root / parts[0] / parts[1]).resolve()
                        if folder.is_relative_to(root) and folder.is_dir():
                            if (
                                event.get("ok") is not True
                                and job.get("status") not in {"completed", "skipped"}
                                and (folder / "intermediate").is_dir()
                            ):
                                folder /= "intermediate"
                            self._result_folder = folder
                            self._result_job = ("/".join(parts), job.get("status"))
                            done = job.get("status") in {"completed", "skipped"}
                            self._review_job = "/".join(parts) if event.get("ok") and done else None
            elif kind == "progress":
                self._state["progress"] = data
            elif kind == "run.started":
                self._state["runtime"] = data.get("runtime")
            elif kind == "stage.started":
                self._state["phase"] = data.get("stage", "processing")
                self._state.pop("stage_progress", None)  # bars never carry over between phases
            elif kind == "stage.finished":
                seconds = data.get("seconds")
                stages = self._state.setdefault("stages", {})
                if isinstance(seconds, (int, float)) and len(stages) < 40:
                    stages[str(data.get("stage"))] = seconds
            elif kind == "stage.progress":
                self._state["stage_progress"] = data
            elif kind == "job.reused":
                self._state["reused"] = data
                self._state["phase"] = "reuse"
            elif kind == "job.prepared":
                self._state["prepared"] = data
            elif kind == "job.finished":
                self._state["summary"] = data
            elif kind in {"warning", "desktop.stage", "log"}:
                self._messages.append(data.get("message", ""))
            if kind == "resource.sample":
                self._state["resources"] = data

    def status(self) -> dict:
        with self._state_lock:
            return json.loads(
                json.dumps(
                    {
                        **self._state,
                        "messages": list(self._messages),
                        "can_open": self._result_folder is not None,
                        "can_review": self._review_job is not None and not self._running,
                    }
                )
            )

    def result_details(self) -> dict:
        """Inspect the finished version folder (called off the UI thread by the page)."""
        with self._state_lock:
            folder, job = self._result_folder, self._result_job
            if folder is None or job is None or self._running:
                raise ValueError("No finished result is available")
        job_id, status = job
        done = status in {"completed", "skipped"}
        details = inspect_files(folder, self._config, job_id, self._processor, partial=not done)
        if self._selected is not None:
            details["queued"] = queued_file_outcome(
                self._config, job_id, self._selected["path"], status
            )
        return details

    def open_result(self) -> dict:
        with self._state_lock:
            folder = self._result_folder
        if folder is None or self._running:
            raise ValueError("No finished result is available")
        open_folder(folder)
        return {"opened": True}

    def can_close(self) -> bool:
        if self._running or self._operation.locked():
            return False
        self.shutdown()  # no FFmpeg or served file may outlive the window
        return True

    def playback(self) -> dict:
        """Cues from the canonical transcript plus the verified media candidates (GUI-09)."""
        with self._state_lock:
            job_id = None if self._running else self._review_job
        if job_id is None or self._selected is None:
            raise ValueError("No finished result is available to review")
        delivery = ArtifactDelivery(self._config, job_id)
        data = json.loads(delivery.existing("transcript.json").read_text(encoding="utf-8"))
        meta = data.get("audio_metadata") or {}
        start = meta.get("selection_start_seconds", 0)
        start = float(start) if isinstance(start, (int, float)) and start >= 0 else 0.0
        cues = [
            {"start": float(cue["start"]), "end": float(cue["end"]), "text": str(cue["text"])}
            for cue in data.get("segments", [])
            if isinstance(cue, dict) and isinstance(cue.get("text"), str)
        ]
        length = float(meta.get("duration_seconds") or (cues[-1]["end"] if cues else 0))
        selected = self._selected
        # A full-length selection plays the whole source; a bounded one stops at its end.
        end = None if start == 0 and length >= selected["duration"] - 0.5 else start + length
        stamp = tuple(selected["identity"][1:])
        queued = selected["path"]
        sources = [
            (queued, stamp, "the queued recording"),
            (
                self._config.processed_dir / job_id.split("/")[0] / queued.name,
                stamp,
                "the archived recording",
            ),
        ]
        if queued.resolve() in self._imported:
            original, size, mtime = self._imported[queued.resolve()]
            sources.append((original, (size, mtime), "the original you imported"))
        found = []
        for path, expected, label in sources:
            try:
                info = path.stat()
            except OSError:
                continue
            if (info.st_size, info.st_mtime_ns) == expected:  # never a different file
                found.append((path, start, end, label, expected))
        for name, label in (
            ("video.mp4", "the 720p preview video"),
            ("audio.mp3", "the exported MP3 audio"),
            ("audio.flac", "the exported FLAC audio"),
        ):
            path = delivery.existing(name)
            if path.is_file():
                found.append((path, 0.0, None, label, None))
        if self._playback_server is None:
            self._playback_server = PlaybackServer()
        self._candidates = found
        self._candidate_serial += 1  # a fresh URL per review: no stale media-cache reuse
        url = self._playback_server.base_url
        return {
            "candidates": [
                {
                    "url": f"{url}?c={i}&n={self._candidate_serial}",
                    "label": label,
                    "offset": offset,
                    "end": stop,
                }
                for i, (_, offset, stop, label, _) in enumerate(found)
            ],
            "cues": cues,
            "selection_start": start,
            "selection_duration": length,
        }

    def playback_select(self, index: int) -> dict:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not (0 <= index < len(self._candidates))
        ):
            raise ValueError("Unknown playback source")
        path, _, _, _, expected = self._candidates[index]
        try:
            info = path.stat()
        except OSError:
            raise ValueError("That recording is no longer available") from None
        if expected is not None and (info.st_size, info.st_mtime_ns) != expected:
            raise ValueError("That recording changed and cannot be played with these subtitles")
        self._playback_server.serve(path)
        return {"index": index}

    def playback_stop(self) -> dict:
        if self._playback_server is not None:
            self._playback_server.release()
        return {"stopped": True}
