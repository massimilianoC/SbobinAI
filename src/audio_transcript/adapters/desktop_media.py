"""Local desktop import and inexpensive, optional waveform preparation."""

from __future__ import annotations

import array
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .ffmpeg import FFmpegProcessor


def import_media(source: Path, queue: Path) -> Path:
    """Copy bytes into a hidden staging directory, then publish without overwriting."""
    source, queue = source.resolve(), queue.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError("Choose a regular media file")
    queue.mkdir(parents=True, exist_ok=True)
    if source.parent == queue:
        return source
    before = source.stat()
    if shutil.disk_usage(queue).free < before.st_size + 16 * 1024 * 1024:
        raise OSError("Not enough free space to import this recording")
    # A hidden directory excludes incomplete imports from both run and watch discovery.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=queue) as scratch:
        staging = Path(scratch) / "media.part"
        with source.open("rb") as original, staging.open("xb") as target:
            shutil.copyfileobj(original, target, 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError("The source changed while being imported; try again when it is stable")
        if staging.stat().st_size != before.st_size:
            raise OSError("The imported copy is incomplete")
        destination = queue / source.name
        if destination.exists():
            destination = queue / f"{source.stem}-{uuid.uuid4().hex[:10]}{source.suffix}"
        # Hard-link publication is atomic and fails rather than overwriting a collision.
        # Both files are on the same volume. A filesystem without hard links fails safely.
        os.link(staging, destination)
    return destination


class WaveformCancelled(Exception):
    """The caller asked for the decode to stop; FFmpeg has exited."""


def _await_exit(process: subprocess.Popen, cancel: threading.Event | None, timeout: float) -> None:
    """Poll about every 100 ms; on cancel or timeout stop FFmpeg and wait for it to exit."""
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                process.wait(timeout=0.1)
                return
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                raise WaveformCancelled
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(process.args, timeout)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def waveform(
    processor: FFmpegProcessor,
    source: Path,
    scratch: Path,
    bins: int = 800,
    cancel: threading.Event | None = None,
) -> dict:
    """Decode low-rate mono PCM to disk; keep at most one waveform bin in memory."""
    duration = processor.probe(source).duration
    if cancel is not None and cancel.is_set():
        raise WaveformCancelled
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="waveform-", dir=scratch) as directory:
        pcm, log = Path(directory) / "preview.pcm", Path(directory) / "ffmpeg.log"
        command = [
            processor.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "2000",
            "-f",
            "s16le",
            str(pcm),
        ]
        # stderr goes to a file so a chatty decoder can never block on an unread pipe;
        # FFmpeg has always exited before the temporary directory is removed.
        with log.open("wb") as errors:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                **hidden_process(),
            )
            _await_exit(process, cancel, 3600)
        if process.returncode:
            detail = log.read_text(encoding="utf-8", errors="replace")[-1200:]
            raise RuntimeError(f"Waveform preparation failed: {detail}")
        samples = pcm.stat().st_size // 2
        if not samples:
            raise ValueError("No audio was decoded")
        # Limit even a very long recording's read buffer to 64 KiB.
        values = []
        with pcm.open("rb") as stream:
            for index in range(bins):
                if cancel is not None and cancel.is_set():
                    raise WaveformCancelled
                count = math.floor((index + 1) * samples / bins) - math.floor(
                    index * samples / bins
                )
                peak = 0
                while count:
                    block = min(count, 32768)
                    data = array.array("h")
                    data.frombytes(stream.read(block * 2))
                    if sys.byteorder == "big":
                        data.byteswap()
                    peak = max(peak, max((abs(n) for n in data), default=0))
                    count -= block
                values.append(peak / 32768)
    return {"duration": duration, "peaks": values}


def hidden_process() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def open_folder(path: Path) -> None:
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("The result folder does not exist")
    if os.name == "nt":
        os.startfile(str(path))
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
