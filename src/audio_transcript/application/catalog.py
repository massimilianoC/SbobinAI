"""Machine-readable ``source.json`` and ``catalog.json`` derived from the on-disk layout.

Both files are rebuilt from ``process/<source>/<version>/metadata.json`` and the matching
output reports, so the same code serves live bookkeeping and the offline ``catalog`` command.
Paths inside these files are relative; absolute machine paths are never written.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import AppConfig
from .layout import parse_version_folder, scope_label
from .state import atomic_json, read_json

SCHEMA_VERSION = 1
CATALOG_NAME = "catalog.json"
SOURCE_NAME = "source.json"
KEY_BASIS_NAME = "source-key.json"
_SOURCE_FOLDER = re.compile(r"^.+-(?P<key>[0-9a-f]{12})$")
_ARTIFACTS = ("transcript.json", "transcript.txt", "report.json", "run-report.md")


def _number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def source_folders(config: AppConfig) -> list[str]:
    """Source folders: directories holding at least one version-named sub-folder."""
    found: set[str] = set()
    for root in (config.process_dir, config.output_dir):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            try:
                children = list(entry.iterdir())
            except OSError:
                continue
            if any(parse_version_folder(child.name) and child.is_dir() for child in children):
                found.add(entry.name)
    return sorted(found)


def version_folders(config: AppConfig, source_folder: str) -> list[str]:
    found: set[str] = set()
    for root in (config.process_dir, config.output_dir):
        try:
            entries = list((root / source_folder).iterdir())
        except OSError:
            continue
        found.update(e.name for e in entries if e.is_dir() and parse_version_folder(e.name))
    return sorted(found)


def _completion(meta: dict, report: dict | None, process_dir: Path, status: str) -> float | None:
    if status == "completed":
        return 100.0
    total = _number(_dict(report).get("total_chunks")) if report else None
    done = _number(_dict(report).get("completed_chunks")) if report else None
    if total is None or done is None:
        total = _number(meta.get("chunk_count"))
        checkpoint = read_json(process_dir / "checkpoint.json", {})
        chunks = _dict(checkpoint).get("chunks")
        done = len(chunks) if isinstance(chunks, dict) else (0 if status == "prepared" else None)
    if total is None or done is None:
        return None
    if total <= 0:
        return None
    return round(100.0 * min(done, total) / total, 1)


def version_summary(config: AppConfig, source_folder: str, version: str) -> dict:
    process_dir = config.process_dir / source_folder / version
    output_dir = config.output_dir / source_folder / version
    meta = _dict(read_json(process_dir / "metadata.json", {}))
    parsed = parse_version_folder(version) or {}
    final_report = read_json(output_dir / "report.json")
    intermediate_report = read_json(output_dir / "intermediate" / "report.json")
    kind = None
    report = None
    base = None
    if isinstance(final_report, dict):
        kind, report, base = "final", final_report, output_dir
    elif isinstance(intermediate_report, dict):
        kind, report, base = "intermediate", intermediate_report, output_dir / "intermediate"
    status = meta.get("status")
    if not isinstance(status, str):
        status = "completed" if kind == "final" else "incomplete"
    execution = _dict(_dict(report).get("execution"))
    chunks = _dict(report)
    configuration = _dict(meta.get("inference_configuration"))
    audio = _dict(meta.get("audio_metadata"))
    scope = meta.get("scope")
    if not isinstance(scope, dict):
        label = parsed.get("scope", "unknown")
        if label == "full":
            scope = {"kind": "full", "max_duration_seconds": None}
        elif label == "unknown":
            scope = {"kind": "unknown", "max_duration_seconds": None}
        else:
            seconds = float(label[len("first") : -1].replace("p", "."))
            scope = {"kind": "bounded", "max_duration_seconds": seconds}
    if "label" not in scope:
        # Legacy (migrated) versions: give every scope the same shape for consumers.
        kind = scope.get("kind")
        seconds = scope.get("max_duration_seconds")
        scope = {
            **scope,
            "label": "unknown" if kind == "unknown" else scope_label(seconds),
        }
    total = _number(execution.get("total_chunks"))
    if total is None:
        total = _number(chunks.get("total_chunks"))
    if total is None:
        total = _number(meta.get("chunk_count"))
    created = meta.get("created_at")
    if not isinstance(created, str):
        created = parsed["created"].isoformat() if parsed else None
    artifacts = {"kind": kind, "transcript": None, "report": None, "run_report": None}
    if base is not None:
        prefix = f"{version}/intermediate/" if kind == "intermediate" else f"{version}/"
        for key, name in (
            ("transcript", "transcript.json"),
            ("report", "report.json"),
            ("run_report", "run-report.md"),
        ):
            if (base / name).is_file():
                artifacts[key] = prefix + name
    return {
        "version": version,
        "created_utc": created,
        "updated_utc": meta.get("updated_at"),
        "status": status,
        "scope": scope,
        "model": meta.get("model") or configuration.get("model") or _dict(report).get("model"),
        "backend": meta.get("backend") or configuration.get("backend"),
        "response_mode": meta.get("response_mode") or configuration.get("response_mode"),
        "language": meta.get("language", configuration.get("language")),
        "completion_percent": _completion(meta, report, process_dir, status),
        "analysed_duration_seconds": _number(
            execution.get("analysed_audio_seconds", audio.get("duration_seconds"))
        ),
        "speech_seconds": _number(audio.get("speech_seconds")),
        "chunks": {
            "total": total,
            "ok": _number(execution.get("chunks_ok")),
            "no_speech": _number(execution.get("chunks_no_speech")),
            "failed": _number(execution.get("chunks_failed")),
        },
        "real_time_factor": _number(execution.get("real_time_factor")),
        "fallbacks": {
            "temperature_fallbacks_used": _number(execution.get("temperature_fallbacks_used")),
            "splits_used": _number(execution.get("splits_used")),
        },
        "stages_seconds": execution.get("stages_seconds")
        if isinstance(execution.get("stages_seconds"), dict)
        else None,
        "confidence": execution.get("confidence")
        if isinstance(execution.get("confidence"), dict)
        else None,
        "artifacts": artifacts,
    }


def _relative(root: Path, path: Path) -> str:
    return f"{root.name}/{path.relative_to(root).as_posix()}"


def _locate(config: AppConfig, source_folder: str, names: list[str], size, hints: list) -> dict:
    processed = config.processed_dir / source_folder
    try:
        archived = sorted(p for p in processed.iterdir() if p.is_file())
    except OSError:
        archived = []
    candidates = list(archived)
    candidates.extend(Path(hint) for hint in hints if isinstance(hint, str) and hint)
    candidates.extend(config.input_dir / name for name in names)
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            if size is not None and candidate.stat().st_size != size:
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        for kind, root in (("processed", config.processed_dir), ("input", config.input_dir)):
            try:
                return {"kind": kind, "path": _relative(root.resolve(), resolved)}
            except ValueError:
                continue
        return {"kind": "external", "path": None}
    return {"kind": "missing", "path": None}


def build_source(config: AppConfig, source_folder: str) -> dict | None:
    versions = version_folders(config, source_folder)
    if not versions:
        return None
    summaries = [version_summary(config, source_folder, version) for version in versions]
    summaries.sort(
        key=lambda item: (item["created_utc"] or "", item["updated_utc"] or "", item["version"]),
        reverse=True,
    )
    names: list[str] = []
    hints: list = []
    size = None
    duration = None
    for version in sorted(versions, reverse=True):
        meta = _dict(read_json(config.process_dir / source_folder / version / "metadata.json", {}))
        name = meta.get("source_name")
        if isinstance(name, str) and name not in names:
            names.append(name)
        hints.extend((meta.get("last_seen_path"), meta.get("archived_source_path")))
        source_meta = _dict(meta.get("source_metadata"))
        if size is None:
            size = _number(source_meta.get("size_bytes", _dict(meta.get("identity")).get("size")))
        if duration is None:
            duration = _number(source_meta.get("duration_seconds"))
    matched = _SOURCE_FOLDER.match(source_folder)
    basis = _dict(read_json(config.process_dir / source_folder / KEY_BASIS_NAME, {}))
    updated = max((item["updated_utc"] or "" for item in summaries), default="") or None
    return {
        "schema_version": SCHEMA_VERSION,
        "folder": source_folder,
        "source_key": matched.group("key") if matched else None,
        "source_key_basis": basis.get("source_key_basis", "content"),
        "original_names": names,
        "size_bytes": size,
        "duration_seconds": duration,
        "location": _locate(config, source_folder, names, size, hints),
        "updated_utc": updated,
        "version_count": len(summaries),
        "versions": summaries,
    }


def write_source(config: AppConfig, source_folder: str) -> dict | None:
    source = build_source(config, source_folder)
    if source is not None:
        atomic_json(config.output_dir / source_folder / SOURCE_NAME, source)
    return source


def write_catalog(config: AppConfig) -> dict:
    """Aggregate every ``source.json`` into ``output/catalog.json``, newest activity first."""
    entries = []
    try:
        folders = sorted(p for p in config.output_dir.iterdir() if p.is_dir())
    except OSError:
        folders = []
    for folder in folders:
        source = read_json(folder / SOURCE_NAME)
        if not isinstance(source, dict) or not source.get("versions"):
            continue
        latest = source["versions"][0]
        names = source.get("original_names") or []
        entries.append(
            {
                "folder": folder.name,
                "source_key": source.get("source_key"),
                "original_name": names[0] if names else None,
                "location": _dict(source.get("location")).get("kind"),
                "version_count": source.get("version_count"),
                "last_activity_utc": source.get("updated_utc"),
                "latest_version": {
                    key: latest.get(key)
                    for key in (
                        "version",
                        "status",
                        "scope",
                        "model",
                        "completion_percent",
                        "created_utc",
                        "updated_utc",
                        "confidence",
                    )
                },
            }
        )
    entries.sort(key=lambda item: (item["last_activity_utc"] or "", item["folder"]), reverse=True)
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "source_count": len(entries),
        "sources": entries,
    }
    atomic_json(config.output_dir / CATALOG_NAME, catalog)
    return catalog


def refresh_source(config: AppConfig, source_folder: str) -> None:
    """Update one source.json and the catalog; used at job start, status changes and end."""
    write_source(config, source_folder)
    write_catalog(config)


def rebuild_all(config: AppConfig) -> dict:
    """Rebuild every source.json and the catalog from disk; no inference, idempotent."""
    count = 0
    versions = 0
    for folder in source_folders(config):
        source = write_source(config, folder)
        if source is not None:
            count += 1
            versions += source["version_count"]
    write_catalog(config)
    return {"sources": count, "versions": versions}
