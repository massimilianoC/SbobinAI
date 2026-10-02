"""Run events: the single data stream behind the console UI, logs and a future web UI.

Every observable step of a run is an :class:`Event`. An :class:`EventBus` stamps events with a
monotonic sequence number and fans them out to pluggable sinks. Emitting never raises into
the caller: a failing sink is counted and reported once on stderr.

Payloads (``data``) are JSON-serializable numbers, labels and short status lines. They never
contain transcript or prompt text, and never absolute paths outside the project roots.
See ``docs/events.md`` for the schema of every event type.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = 1

# Event types, for reference and validation in consumers.
EVENT_TYPES = (
    "run.started",
    "job.started",
    "stage.started",
    "stage.finished",
    "job.prepared",
    "chunk.started",
    "chunk.attempt",
    "chunk.finished",
    "progress",
    "resource.sample",
    "log",
    "warning",
    "job.finished",
    "run.finished",
)

STATUS_WRITE_INTERVAL = 0.5


def new_run_id(now: datetime | None = None) -> str:
    """``YYYYMMDDTHHMMSSZ-<4 hex>``."""
    moment = now or datetime.now(UTC)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"


def utc_iso(moment: datetime | None = None) -> str:
    return (moment or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True)
class Event:
    seq: int
    ts: str
    elapsed_s: float
    run_id: str | None
    type: str
    data: dict
    job_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "seq": self.seq,
            "ts": self.ts,
            "elapsed_s": self.elapsed_s,
            "run_id": self.run_id,
        }
        if self.job_id is not None:
            record["job_id"] = self.job_id
        record["type"] = self.type
        record["data"] = self.data
        return record

    def to_json(self) -> str:
        record = self.to_dict()
        try:
            return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError):
            return json.dumps(
                _clean(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


class Sink(Protocol):
    def handle(self, event: Event) -> None: ...

    def close(self) -> None: ...


class NullBus:
    """Default emitter: does nothing, has no run id."""

    run_id: str | None = None
    sink_errors = 0

    def emit(self, type_: str, data: dict | None = None, *, job_id: str | None = None) -> None:
        return None

    def add_sink(self, sink: Sink) -> None:
        return None

    def close(self) -> None:
        return None


NULL_EVENTS = NullBus()


class EventBus:
    """Thread-safe fan-out of events to sinks; never raises from :meth:`emit`."""

    def __init__(
        self,
        run_id: str | None = None,
        sinks: list[Sink] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        stderr=None,
    ):
        self.run_id = run_id or new_run_id()
        self._sinks: list[Sink] = list(sinks or [])
        self._clock = clock
        self._started = clock()
        self._seq = 0
        self._lock = threading.RLock()
        self._stderr = stderr
        self._reported: set[int] = set()
        self.sink_errors = 0
        self._closed = False

    def add_sink(self, sink: Sink) -> None:
        with self._lock:
            self._sinks.append(sink)

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    def emit(self, type_: str, data: dict | None = None, *, job_id: str | None = None) -> None:
        try:
            with self._lock:
                if self._closed:
                    return
                self._seq += 1
                event = Event(
                    seq=self._seq,
                    ts=utc_iso(),
                    elapsed_s=round(self._clock() - self._started, 3),
                    run_id=self.run_id,
                    type=type_,
                    data=dict(data or {}),
                    job_id=job_id,
                )
                for sink in self._sinks:
                    try:
                        sink.handle(event)
                    except Exception as exc:
                        self._sink_failed(sink, exc)
        except Exception:  # pragma: no cover - the bus itself must never break a run
            self.sink_errors += 1

    def _sink_failed(self, sink: Sink, exc: Exception) -> None:
        self.sink_errors += 1
        key = id(sink)
        if key in self._reported:
            return
        self._reported.add(key)
        stream = self._stderr or sys.stderr
        try:
            print(
                f"WARNING: event sink {type(sink).__name__} failed "
                f"({type(exc).__name__}: {exc}); further errors of this sink are not shown",
                file=stream,
            )
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for sink in self._sinks:
                try:
                    sink.close()
                except Exception as exc:
                    self._sink_failed(sink, exc)


# ---------------------------------------------------------------------------- file helpers


def _atomic_text(path: Path, text: str) -> None:
    """Replace ``path`` atomically without fsync (status snapshots are disposable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:  # a reader holds the file open on Windows
                if attempt == 3:
                    raise
                time.sleep(0.02)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class LineFile:
    """Append-only, line-buffered text file flushed after every line."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8", newline="\n", buffering=1)

    def write(self, line: str) -> None:
        self._stream.write(line + "\n")
        self._stream.flush()

    def close(self) -> None:
        try:
            self._stream.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------- sinks


@dataclass
class RunPaths:
    process_dir: Path
    run_id: str
    run_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.run_dir = self.process_dir / "runs" / self.run_id

    @property
    def events(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def status(self) -> Path:
        return self.run_dir / "status.json"

    @property
    def log(self) -> Path:
        return self.run_dir / "pipeline.log"

    @property
    def latest(self) -> Path:
        return self.process_dir / "runs" / "latest.json"

    def relative(self, path: Path) -> str:
        return path.relative_to(self.process_dir).as_posix()


class RunStreamSink:
    """``events.jsonl`` + throttled ``status.json`` + ``latest.json`` of one run."""

    def __init__(self, paths: RunPaths, state, *, clock: Callable[[], float] = time.monotonic):
        self.paths = paths
        self.state = state
        self._clock = clock
        self._file = LineFile(paths.events)
        self._last_status = -1e9
        self._dirty = False
        self._write_latest("running", utc_iso())

    def _write_latest(self, state: str, started: str) -> None:
        pointer = {
            "run_id": self.paths.run_id,
            "state": state,
            "started_at": started,
            "updated_at": utc_iso(),
            "process_dir": str(self.paths.process_dir.resolve()),
            "paths": {
                "run_dir": self.paths.relative(self.paths.run_dir),
                "events": self.paths.relative(self.paths.events),
                "status": self.paths.relative(self.paths.status),
                "log": self.paths.relative(self.paths.log),
            },
        }
        self._started = started
        _atomic_text(self.paths.latest, json.dumps(pointer, indent=2) + "\n")

    def _write_status(self) -> None:
        _atomic_text(self.paths.status, json.dumps(_clean(self.state.snapshot()), indent=2) + "\n")
        self._last_status = self._clock()
        self._dirty = False

    def handle(self, event: Event) -> None:
        self._file.write(event.to_json())
        self._dirty = True
        important = event.type.startswith(("run.", "job.", "stage."))
        if important or self._clock() - self._last_status >= STATUS_WRITE_INTERVAL:
            self._write_status()
        if event.type == "run.finished":
            self._write_latest("finished", self._started)

    def close(self) -> None:
        try:
            if self._dirty:
                self._write_status()
        finally:
            self._file.close()


class JobStreamSink:
    """Copies each job's events to ``process/<source>/<version>/events.jsonl``.

    The file is appended to across runs, so every version keeps its own history; each line
    carries its ``run_id``. Resource samples are run-level and are not copied.
    """

    def __init__(self, process_dir: Path):
        self.process_dir = process_dir
        self._files: dict[str, LineFile] = {}

    def handle(self, event: Event) -> None:
        if event.job_id is None or event.type == "resource.sample":
            return
        stream = self._files.get(event.job_id)
        if stream is None:
            folder, _, version = event.job_id.partition("/")
            stream = LineFile(self.process_dir / folder / version / "events.jsonl")
            self._files[event.job_id] = stream
        stream.write(event.to_json())
        if event.type == "job.finished":
            self._files.pop(event.job_id).close()

    def close(self) -> None:
        for stream in self._files.values():
            stream.close()
        self._files.clear()


class PlainLogSink:
    """``pipeline.log``: timestamped status lines, warnings and the final summary.

    Always written, whatever the console mode, because native-process output is not captured
    by wrappers such as PowerShell ``Start-Transcript``.
    """

    def __init__(self, path: Path, summary: Callable[[], list[str]] | None = None):
        self._file = LineFile(path)
        self._summary = summary

    def handle(self, event: Event) -> None:
        stamp = event.ts
        if event.type == "log":
            level = str(event.data.get("level", "info"))
            self._file.write(f"{stamp} [{level}] {event.data.get('message', '')}")
        elif event.type == "warning":
            self._file.write(f"{stamp} [warning] {event.data.get('message', '')}")
        elif event.type == "run.started":
            data = event.data
            self._file.write(
                f"{stamp} [run] started {event.run_id}: command={data.get('command')} "
                f"model={data.get('model')} backend={data.get('backend')} "
                f"queue={data.get('queue_size')}"
            )
        elif event.type == "run.finished":
            if self._summary is not None:
                for line in self._summary():
                    self._file.write(f"{stamp} [summary] {line}")
            self._file.write(
                f"{stamp} [run] finished: exit={event.data.get('exit_status')} "
                f"wall={event.data.get('wall_seconds')}s"
            )

    def close(self) -> None:
        self._file.close()


class JsonlStdoutSink:
    """``--ui jsonl``: one JSON event per stdout line, for piping into other tools."""

    def __init__(self, stream=None):
        self._stream = stream

    def handle(self, event: Event) -> None:
        stream = self._stream or sys.stdout
        stream.write(event.to_json() + "\n")
        stream.flush()

    def close(self) -> None:
        return None


class CallbackSink:
    def __init__(self, callback: Callable[[Event], None]):
        self._callback = callback

    def handle(self, event: Event) -> None:
        self._callback(event)

    def close(self) -> None:
        return None
