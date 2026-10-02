"""Pure naming rules for the per-source, per-version folder layout.

``process/<source-folder>/<version-folder>/`` and ``output/<source-folder>/<version-folder>/``
mirror each other. A source folder is ``<safe-stem>-<source key>``; the source key identifies
the media bytes and therefore survives rename, move to ``processed`` and mtime changes. A
version folder is ``<created UTC>_<model slug>_<scope>_<fingerprint prefix>`` where the
fingerprint prefix is the stable identity of the configuration.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from pathlib import Path

SOURCE_KEY_LENGTH = 12
SOURCE_KEY_BLOCK = 4 * 1024 * 1024
FINGERPRINT_PREFIX_LENGTH = 8
VERSION_TIME_FORMAT = "%Y%m%dT%H%MZ"

_VERSION_PATTERN = re.compile(
    r"^(?P<created>\d{8}T\d{4}Z)_(?P<model>[a-z0-9.\-]+)_"
    r"(?P<scope>full|unknown|first\d+(?:p\d+)?s)_(?P<fp>[0-9a-f]{8})$"
)


class LayoutError(RuntimeError):
    """The folder layout is ambiguous or inconsistent; nothing was changed."""


def source_key_from_stream(stream, size: int) -> str:
    """Hash the size plus the first and last 4 MiB of an already open binary stream."""
    digest = hashlib.sha256(f"{size}:".encode())
    digest.update(stream.read(SOURCE_KEY_BLOCK))
    stream.seek(max(0, size - SOURCE_KEY_BLOCK))
    digest.update(stream.read(SOURCE_KEY_BLOCK))
    return digest.hexdigest()[:SOURCE_KEY_LENGTH]


def source_key(path: Path) -> str:
    """Cheap identity of media bytes: two reads of at most 4 MiB, whatever the file size."""
    with Path(path).open("rb") as stream:
        size = Path(path).stat().st_size
        return source_key_from_stream(stream, size)


def metadata_source_key(size: int, name: str) -> str:
    """Fallback key for a source that no longer exists (derived from recorded size and name)."""
    return hashlib.sha256(f"metadata:{size}:{name}".encode()).hexdigest()[:SOURCE_KEY_LENGTH]


def safe_stem(file_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file_name).stem).strip("._-") or "media"
    return stem[:64]


def source_folder_name(file_name: str, key: str) -> str:
    return f"{safe_stem(file_name)}-{key}"


def model_slug(model: str | None) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", (model or "").lower()).strip("-.")
    return slug[:40].strip("-.") or "unknown"


def _number_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").replace(".", "p")


def scope_label(max_duration: float | None) -> str:
    if max_duration is None:
        return "full"
    if not math.isfinite(max_duration) or max_duration <= 0:
        return "unknown"
    return f"first{_number_label(max_duration)}s"


def fingerprint_prefix(fingerprint: str) -> str:
    return fingerprint[:FINGERPRINT_PREFIX_LENGTH]


def version_timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(VERSION_TIME_FORMAT)


def version_folder_name(
    created: datetime, model: str | None, max_duration: float | None, fingerprint: str
) -> str:
    return (
        f"{version_timestamp(created)}_{model_slug(model)}_"
        f"{scope_label(max_duration)}_{fingerprint_prefix(fingerprint)}"
    )


def parse_version_folder(name: str) -> dict | None:
    """Split a version folder name into its parts, or None when it is not one."""
    matched = _VERSION_PATTERN.match(name)
    if matched is None:
        return None
    created = datetime.strptime(matched.group("created"), VERSION_TIME_FORMAT).replace(tzinfo=UTC)
    return {
        "created": created,
        "model_slug": matched.group("model"),
        "scope": matched.group("scope"),
        "fingerprint_prefix": matched.group("fp"),
    }


def find_source_folder(roots: list[Path], key: str) -> str | None:
    """Reuse the existing folder of a source key even if the file was renamed since."""
    suffix = f"-{key}"
    found: set[str] = set()
    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        found.update(e.name for e in entries if e.name.endswith(suffix) and e.is_dir())
    if len(found) > 1:
        raise LayoutError(
            f"More than one source folder carries key {key}: "
            + ", ".join(sorted(found))
            + ". Merge them manually."
        )
    return next(iter(found), None)


def find_version_folder(roots: list[Path], source_folder: str, fingerprint: str) -> str | None:
    """Find the single existing version folder for a fingerprint inside a source folder.

    Every root (process, output) is searched so a half-written layout is still found.
    More than one distinct match is an error: the layout must not guess.
    """
    prefix = fingerprint_prefix(fingerprint)
    found: set[str] = set()
    for root in roots:
        directory = root / source_folder
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            parsed = parse_version_folder(entry.name)
            if parsed is not None and parsed["fingerprint_prefix"] == prefix and entry.is_dir():
                found.add(entry.name)
    if len(found) > 1:
        raise LayoutError(
            f"More than one version folder matches fingerprint {prefix} in {source_folder}: "
            + ", ".join(sorted(found))
            + ". Remove or rename the duplicate manually."
        )
    return next(iter(found), None)
