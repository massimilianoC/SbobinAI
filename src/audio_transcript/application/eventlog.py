"""Reading and following a run's ``events.jsonl`` (the ``events`` command).

Consumers such as a future web UI can use the same files: read ``status.json`` to poll, or
tail ``events.jsonl`` for every event.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from .state import read_json


def latest_run_id(process_dir: Path) -> str | None:
    pointer = read_json(process_dir / "runs" / "latest.json")
    run_id = pointer.get("run_id") if isinstance(pointer, dict) else None
    return run_id if isinstance(run_id, str) else None


def events_path(process_dir: Path, run: str = "latest") -> Path | None:
    run_id = latest_run_id(process_dir) if run == "latest" else run
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return None
    return process_dir / "runs" / run_id / "events.jsonl"


def _matches(line: str, prefix: str | None) -> tuple[bool, str | None]:
    try:
        record = json.loads(line)
    except ValueError:
        return False, None
    kind = record.get("type") if isinstance(record, dict) else None
    if not isinstance(kind, str):
        return False, None
    return (prefix is None or kind.startswith(prefix)), kind


def follow_events(
    path: Path,
    emit: Callable[[str], None],
    *,
    type_prefix: str | None = None,
    follow: bool = False,
    poll_interval: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    max_polls: int | None = None,
) -> int:
    """Print events as JSONL; with ``follow``, keep polling until ``run.finished``.

    Returns ``0`` normally and ``1`` when the file does not exist and ``follow`` is off.
    ``max_polls`` bounds the waiting loop (tests); ``None`` waits until the run finishes.
    """
    offset = 0
    buffer = b""
    polls = 0
    while True:
        finished = False
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset += len(chunk)
        except FileNotFoundError:
            if not follow:
                return 1
            chunk = b""
        buffer += chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            show, kind = _matches(line, type_prefix)
            if show:
                emit(line)
            if kind == "run.finished":
                finished = True
        if not follow or finished:
            return 0
        polls += 1
        if max_polls is not None and polls >= max_polls:
            return 0
        sleep(poll_interval)
