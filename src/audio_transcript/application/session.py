"""One observable run: the event bus with its file sinks, console sink and resource monitor."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

from ..config import AppConfig
from .events import (
    EventBus,
    JobStreamSink,
    JsonlStdoutSink,
    PlainLogSink,
    RunPaths,
    RunStreamSink,
    Sink,
    new_run_id,
)
from .progress import RunState

JOB_OK = {"completed", "skipped", "prepared"}


def summary_lines(state: RunState) -> list[str]:
    """Plain-text run summary (names, counts, timings; never transcript text)."""
    lines = []
    for job in state.jobs:
        counts = job.get("counts") or {}
        timings = job.get("timings") or {}
        confidence = job.get("confidence")
        parts = [f"{job.get('status')}", f"{job.get('source') or job.get('job_id') or '?'}"]
        if counts.get("total_chunks") is not None:
            parts.append(
                f"chunks ok={counts.get('ok')} no_speech={counts.get('no_speech')} "
                f"failed={counts.get('failed')} of {counts.get('total_chunks')}"
            )
        if confidence is not None:
            parts.append(f"confidence {confidence}")
        if timings.get("wall_seconds") is not None:
            parts.append(f"wall {timings['wall_seconds']}s")
        if job.get("error"):
            parts.append(f"error: {job['error']}")
        lines.append(" | ".join(parts))
    if not lines:
        lines.append("no jobs were processed")
    return lines


class RunSession:
    """Owns the bus of one run. ``status_out`` is where plain status lines go."""

    def __init__(
        self,
        config: AppConfig,
        *,
        command: str,
        ui_mode: str = "plain",
        queue_size: int | None = None,
        console_sink: Sink | None = None,
        monitor_factory: Callable[[EventBus], object] | None = None,
        status_out: Callable[[str], None] | None = print,
        stdout=None,
        stderr=None,
        run_id: str | None = None,
        runtime: dict | None = None,
    ):
        self.config = config
        self.runtime = dict(runtime) if runtime else None
        self.command = command
        self.ui_mode = ui_mode
        self.queue_size = queue_size
        self.state = RunState()
        self.run_id = run_id or new_run_id()
        self.paths = RunPaths(config.process_dir, self.run_id)
        self.stderr = stderr if stderr is not None else sys.stderr
        self.bus = EventBus(self.run_id, stderr=self.stderr)
        self.console_sink = console_sink
        self._monitor_factory = monitor_factory
        self._monitor = None
        # Plain mode prints the status lines exactly as before; live and jsonl keep stdout
        # for the renderer / the event stream and rely on the log events.
        self.status_out: Callable[[str], None] = (
            status_out if ui_mode == "plain" and status_out is not None else _discard
        )
        self._stdout = stdout
        self._started = time.monotonic()
        self._finished = False
        self._opened = False

    # -- lifecycle
    def start(self) -> RunSession:
        if self._opened:
            return self
        self._opened = True
        self._add("state", _StateSink(self.state))
        try:
            self.bus.add_sink(RunStreamSink(self.paths, self.state))
        except OSError as exc:
            self._warn(f"run stream unavailable ({exc}); status.json and events.jsonl disabled")
        try:
            self.bus.add_sink(JobStreamSink(self.config.process_dir))
        except OSError as exc:  # pragma: no cover - constructor does not touch the disk
            self._warn(f"per-job event files unavailable ({exc})")
        try:
            self.bus.add_sink(PlainLogSink(self.paths.log, lambda: summary_lines(self.state)))
        except OSError as exc:
            self._warn(f"pipeline.log unavailable ({exc})")
        if self.ui_mode == "jsonl":
            self.bus.add_sink(JsonlStdoutSink(self._stdout))
        if self.console_sink is not None:
            self.bus.add_sink(self.console_sink)
        config = self.config
        self.bus.emit(
            "run.started",
            {
                "command": self.command,
                "model": config.model,
                "backend": config.backend,
                "response_mode": config.response_mode if config.backend == "llamacpp" else None,
                "language": config.language,
                "parallel_requests": config.parallel_requests,
                "queue_size": self.queue_size,
                "ui": self.ui_mode,
                **({"runtime": self.runtime} if self.runtime else {}),
            },
        )
        if self._monitor_factory is not None and config.monitor:
            try:
                self._monitor = self._monitor_factory(self.bus)
                self._monitor.start()
            except Exception as exc:
                self._monitor = None
                self.bus.emit(
                    "warning",
                    {"code": "monitor_unavailable", "message": f"Resource monitor disabled: {exc}"},
                )
        return self

    def _add(self, _name: str, sink: Sink) -> None:
        self.bus.add_sink(sink)

    def _warn(self, message: str) -> None:
        try:
            print(f"WARNING: {message}", file=self.stderr)
        except Exception:
            pass

    def finish(self, exit_status: int, results: list[dict] | None = None) -> None:
        """Emit ``run.finished`` once and stop the monitor."""
        if self._finished or not self._opened:
            return
        self._finished = True
        if self._monitor is not None:
            try:
                self._monitor.stop()
            except Exception:
                pass
        jobs = [
            {
                "job_id": result.get("job_id"),
                "source": Path(str(result["source"])).name if result.get("source") else None,
                "status": result.get("status"),
            }
            for result in (results or [])
            if isinstance(result, dict)
        ]
        self.bus.emit(
            "run.finished",
            {
                "exit_status": exit_status,
                "jobs": jobs,
                "wall_seconds": round(time.monotonic() - self._started, 3),
            },
        )

    def close(self) -> None:
        if self._opened and not self._finished:
            self.finish(1)
        if self._monitor is not None:
            try:
                self._monitor.stop()
            except Exception:
                pass
        self.bus.close()

    def __enter__(self) -> RunSession:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and not self._finished:
            self.finish(130 if exc_type is KeyboardInterrupt else 1)
        self.close()


def exit_status_of(results: list[dict]) -> int:
    return 1 if any(r.get("status") in {"failed", "incomplete"} for r in results) else 0


class _StateSink:
    def __init__(self, state: RunState):
        self._state = state

    def handle(self, event) -> None:
        self._state.apply(event)

    def close(self) -> None:
        return None


def _discard(_message: str) -> None:
    return None
