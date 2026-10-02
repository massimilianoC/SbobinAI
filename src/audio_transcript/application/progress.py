"""Progress accounting and the run state reduced from events.

``ProgressTracker`` turns chunk completions into the numbers of a ``progress`` event.
``RunState`` folds an event stream into one snapshot object; it feeds ``status.json`` and the
console renderer, so both always agree.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from .events import Event

STAGE_SEQUENCE = (
    "probe",
    "audio_extraction",
    "speech_detection",
    "chunk_slicing",
    "inference",
    "export",
)
HISTORY_LENGTH = 60
ACTIVITY_LENGTH = 40
RESOURCE_KEYS = (
    "cpu_percent",
    "ram_used_mb",
    "ram_total_mb",
    "process_rss_mb",
    "gpu_util_percent",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_power_w",
    "gpu_temp_c",
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


class ProgressTracker:
    """Counters over chunk outcomes; thread-safe because in-flight tracking is shared."""

    def __init__(
        self,
        chunk_audio: dict[int, float],
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.total_chunks = len(chunk_audio)
        self.audio_total = sum(chunk_audio.values())
        self._chunk_audio = chunk_audio
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._in_flight: set[int] = set()
        self.ok = self.no_speech = self.failed = 0
        self.fallbacks = self.splits = 0
        self.tokens = 0.0
        self.audio_done = 0.0
        self._run_audio = 0.0
        self._conf_total = self._conf_weight = 0.0

    # -- bookkeeping
    def _entries(self, history: list) -> None:
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if entry.get("action") == "split":
                if entry.get("outcome") == "ok":
                    self.splits += 1
                continue
            if entry.get("stage") == "temperature":
                self.fallbacks += 1
            metrics = entry.get("metrics")
            if not isinstance(metrics, dict):
                continue
            tokens = _number(metrics.get("completion_tokens"))
            if tokens is not None:
                self.tokens += tokens
            mean = _number(metrics.get("mean_token_prob"))
            if mean is not None and entry.get("outcome") == "ok":
                count = _number(metrics.get("logprob_tokens"))
                count = count if count is not None and count > 0 else 1.0
                self._conf_total += mean * count
                self._conf_weight += count

    def seed(self, done: dict, attempts: dict) -> None:
        """Account for chunks already completed in an earlier run (resume)."""
        with self._lock:
            for key, segments in done.items():
                if segments:
                    self.ok += 1
                else:
                    self.no_speech += 1
                self.audio_done += self._chunk_audio.get(int(key), 0.0)
                history = attempts.get(key)
                if isinstance(history, list):
                    self._entries(history)
            self._started = self._clock()

    def begin(self, index: int) -> list[int]:
        with self._lock:
            self._in_flight.add(index)
            return sorted(self._in_flight)

    def end(self, index: int) -> None:
        with self._lock:
            self._in_flight.discard(index)

    def in_flight(self) -> list[int]:
        with self._lock:
            return sorted(self._in_flight)

    def finish_chunk(self, index: int, segments: list | None, history: list) -> str:
        """Record one chunk result and return ``ok``, ``no_speech`` or ``failed``."""
        with self._lock:
            if segments is None:
                outcome = "failed"
                self.failed += 1
            elif segments:
                outcome = "ok"
                self.ok += 1
            else:
                outcome = "no_speech"
                self.no_speech += 1
            audio = self._chunk_audio.get(index, 0.0)
            self.audio_done += audio
            self._run_audio += audio
            self._entries(history)
            return outcome

    # -- reporting
    def counters(self) -> dict:
        with self._lock:
            return {
                "ok": self.ok,
                "no_speech": self.no_speech,
                "failed": self.failed,
                "fallbacks": self.fallbacks,
                "splits": self.splits,
            }

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(0.0, self._clock() - self._started)
            done = self.ok + self.no_speech + self.failed
            total = self.total_chunks
            throughput = self._run_audio / elapsed if elapsed > 0 and self._run_audio > 0 else None
            remaining = max(0.0, self.audio_total - self.audio_done)
            eta = None
            if throughput:
                eta = 0.0 if remaining <= 0 else remaining / throughput
            return {
                "done_chunks": done,
                "total_chunks": total,
                "percent": round(100.0 * done / total, 1) if total else 100.0,
                "audio_done_s": round(self.audio_done, 3),
                "audio_total_s": round(self.audio_total, 3),
                "elapsed_s": round(elapsed, 3),
                "eta_s": None if eta is None else round(eta, 1),
                "real_time_factor": round(1.0 / throughput, 4) if throughput else None,
                "throughput_x": round(throughput, 2) if throughput else None,
                "tokens": self.tokens,
                "ok": self.ok,
                "no_speech": self.no_speech,
                "failed": self.failed,
                "fallbacks": self.fallbacks,
                "splits": self.splits,
                "confidence_mean": (
                    round(self._conf_total / self._conf_weight, 4) if self._conf_weight else None
                ),
                "in_flight": sorted(self._in_flight),
            }


class JobState:
    def __init__(self, job_id: str | None, source_name: str | None):
        self.job_id = job_id
        self.source_name = source_name
        self.version: str | None = None
        self.scope: str | None = None
        self.index = 0
        self.total: int | None = None
        self.status = "running"
        self.stages: dict[str, dict] = {}
        self.prepared: dict | None = None
        self.progress: dict | None = None
        self.in_flight: list[int] = []
        self.current_stage: str | None = None

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_name": self.source_name,
            "version": self.version,
            "scope": self.scope,
            "index": self.index,
            "total": self.total,
            "status": self.status,
            "stage": self.current_stage,
            "stages": self.stages,
            "prepared": self.prepared,
            "progress": self.progress,
        }


class RunState:
    """State object updated from events; ``snapshot()`` is the ``status.json`` content."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.started_at: str | None = None
        self.updated_at: str | None = None
        self.state = "running"
        self.elapsed_s = 0.0
        self.command: str | None = None
        self.model: str | None = None
        self.backend: str | None = None
        self.response_mode: str | None = None
        self.language: str | None = None
        self.parallel_requests: int | None = None
        self.queue_size: int | None = None
        self.exit_status: int | None = None
        self.job: JobState | None = None
        self.jobs: list[dict] = []
        self.jobs_started = 0
        self.resources: dict = {}
        self.history: dict[str, deque] = {
            key: deque(maxlen=HISTORY_LENGTH) for key in RESOURCE_KEYS
        }
        self.activity: deque = deque(maxlen=ACTIVITY_LENGTH)
        self.warnings = 0
        self.last_seq = 0
        self._lock = threading.RLock()

    # -- reducer
    def apply(self, event: Event) -> None:
        with self._lock:
            self._apply(event)

    def tick(self, elapsed_s: float) -> None:
        with self._lock:
            self.elapsed_s = max(self.elapsed_s, elapsed_s)

    def _apply(self, event: Event) -> None:
        data = event.data
        self.run_id = event.run_id or self.run_id
        self.updated_at = event.ts
        self.last_seq = event.seq
        self.elapsed_s = max(self.elapsed_s, event.elapsed_s)
        kind = event.type
        if kind == "run.started":
            self.started_at = event.ts
            for key in (
                "command",
                "model",
                "backend",
                "response_mode",
                "language",
                "parallel_requests",
                "queue_size",
            ):
                setattr(self, key, data.get(key))
        elif kind == "job.started":
            self.jobs_started += 1
            self.job = JobState(event.job_id, data.get("source"))
            self.job.version = data.get("version")
            self.job.scope = data.get("scope")
            self.job.index = int(data.get("index") or self.jobs_started)
            self.job.total = data.get("total")
        elif kind in ("stage.started", "stage.finished") and self.job is not None:
            name = str(data.get("stage"))
            stage = self.job.stages.setdefault(name, {"state": "pending", "seconds": None})
            if kind == "stage.started":
                stage["state"] = "running"
                self.job.current_stage = name
            else:
                stage["state"] = "done"
                stage["seconds"] = data.get("seconds")
                if self.job.current_stage == name:
                    self.job.current_stage = None
        elif kind == "job.prepared" and self.job is not None:
            self.job.prepared = dict(data)
        elif kind == "progress" and self.job is not None:
            self.job.progress = dict(data)
            self.job.in_flight = list(data.get("in_flight") or [])
        elif kind == "chunk.started" and self.job is not None:
            self.job.in_flight = list(data.get("in_flight") or [])
        elif kind == "resource.sample":
            self._resources(data)
        elif kind in ("log", "warning"):
            level = str(data.get("level", "warning" if kind == "warning" else "info"))
            if level != "info":
                self.warnings += 1
            self.activity.append(
                {
                    "ts": event.ts,
                    "elapsed_s": event.elapsed_s,
                    "level": level,
                    "text": str(data.get("message", "")),
                }
            )
        elif kind == "job.finished":
            summary = {
                "job_id": event.job_id,
                "source": data.get("source"),
                "status": data.get("status"),
                "counts": data.get("counts"),
                "timings": data.get("timings"),
                "confidence": data.get("confidence"),
                "artifacts": data.get("artifacts"),
                "error": data.get("error"),
            }
            self.jobs.append(summary)
            if self.job is not None:
                self.job.status = str(data.get("status"))
                self.job.current_stage = None
        elif kind == "run.finished":
            self.state = "finished"
            self.exit_status = data.get("exit_status")

    def _resources(self, data: dict) -> None:
        self.resources = {key: data.get(key) for key in RESOURCE_KEYS}
        self.resources["source"] = data.get("source")
        self.resources["scope"] = data.get("scope")
        for key in RESOURCE_KEYS:
            value = _number(data.get(key))
            if value is not None:
                self.history[key].append(value)

    # -- views
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": 1,
                "run_id": self.run_id,
                "state": self.state,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "elapsed_s": round(self.elapsed_s, 3),
                "last_seq": self.last_seq,
                "command": self.command,
                "model": self.model,
                "backend": self.backend,
                "response_mode": self.response_mode,
                "language": self.language,
                "parallel_requests": self.parallel_requests,
                "queue_size": self.queue_size,
                "exit_status": self.exit_status,
                "current_job": None if self.job is None else self.job.snapshot(),
                "jobs": list(self.jobs),
                "resources": dict(self.resources),
                "warnings": self.warnings,
                "activity": list(self.activity)[-8:],
            }
