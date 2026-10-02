"""Resumable, verified file download using only the standard library."""

from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

CHUNK_BYTES = 1024 * 1024
USER_AGENT = "audio-transcript-setup"

# progress(done_bytes, total_bytes, bytes_per_second)
Progress = Callable[[int, int, float], None]


class DownloadError(RuntimeError):
    """The download failed or the result did not match its pinned identity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(
    url: str,
    destination: Path,
    *,
    sha256: str,
    size_bytes: int,
    progress: Progress | None = None,
    timeout: float = 60.0,
) -> None:
    """Download ``url`` to ``destination`` through ``<destination>.part``.

    An existing ``.part`` file is resumed with an HTTP Range request. The result is
    accepted only when its size and SHA-256 match; then it is renamed atomically.
    A short download keeps the ``.part`` file so the next run resumes it. A complete
    download with a wrong hash deletes the ``.part`` file, because resuming corrupt
    data cannot succeed. An existing ``destination`` is never overwritten.
    """
    if destination.exists():
        raise DownloadError(f"Refusing to overwrite an existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > size_bytes:
        part.unlink()
        offset = 0
    if offset < size_bytes:
        _fetch(url, part, offset, size_bytes, progress, timeout)
    actual_size = part.stat().st_size
    if actual_size != size_bytes:
        raise DownloadError(
            f"Incomplete download ({actual_size} of {size_bytes} bytes); "
            f"run setup again to resume {part.name}"
        )
    actual_hash = sha256_file(part)
    if actual_hash != sha256.lower():
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"SHA-256 mismatch for {destination.name}: expected {sha256.lower()}, got "
            f"{actual_hash}. The partial file was removed; nothing was installed."
        )
    os.replace(part, destination)


def _fetch(
    url: str, part: Path, offset: int, total: int, progress: Progress | None, timeout: float
) -> None:
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            part.unlink(missing_ok=True)
            raise DownloadError(
                "The server rejected the resume range; the partial file was removed. "
                "Run setup again."
            ) from exc
        raise DownloadError(f"HTTP {exc.code} while downloading {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DownloadError(f"Could not reach {url}: {exc}") from exc
    with response:
        status = getattr(response, "status", None) or response.getcode()
        resumed = offset > 0 and status == 206
        if offset and not resumed:
            offset = 0  # the server ignored the Range header and sends the whole file
        done = offset
        started = time.monotonic()
        base = offset
        try:
            with part.open("ab" if resumed else "wb") as out:
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if progress is not None:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        progress(done, total, (done - base) / elapsed)
        except (OSError, http.client.HTTPException) as exc:
            raise DownloadError(f"Download interrupted: {exc}") from exc
