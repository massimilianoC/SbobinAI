"""Move legacy flat job folders into the per-source, per-version layout.

Only explicit ``os.rename`` moves are performed: nothing is deleted, an existing target is
never overwritten, and every move is written to ``process/layout-migration-<UTC>.json``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..config import AppConfig
from .catalog import KEY_BASIS_NAME, rebuild_all
from .layout import (
    find_source_folder,
    metadata_source_key,
    parse_version_folder,
    source_folder_name,
    source_key,
    version_folder_name,
)
from .state import atomic_json, read_json

BOUNDED_TOLERANCE = 0.05


@dataclass
class Move:
    root: str  # "process" | "output" | "processed"
    source: str  # relative to the root
    target: str  # relative to the root
    kind: str = "dir"  # "dir" | "file"


@dataclass
class PlanItem:
    old: str
    status: str  # "move" | "skipped" | "conflict" | "already-migrated"
    reason: str = ""
    source_folder: str = ""
    version_folder: str = ""
    key_basis: str = ""
    moves: list[Move] = field(default_factory=list)


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _roots(config: AppConfig) -> dict[str, Path]:
    return {
        "process": config.process_dir,
        "output": config.output_dir,
        "processed": config.processed_dir,
    }


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _model(meta: dict, output_job: Path) -> str | None:
    model = _dict(meta.get("inference_configuration")).get("model") or meta.get("model")
    if isinstance(model, str) and model:
        return model
    for candidate in (output_job / "transcript.json", output_job / "intermediate/transcript.json"):
        value = _dict(read_json(candidate, {})).get("model")
        if isinstance(value, str) and value:
            return value
    return None


def _scope_seconds(meta: dict, job: Path) -> float | None:
    """Analysed seconds for a bounded run, ``math.inf`` for a full run, None when unknown."""
    audio = _number(_dict(meta.get("audio_metadata")).get("duration_seconds"))
    source = _number(_dict(meta.get("source_metadata")).get("duration_seconds"))
    if audio is None or source is None:
        prepared = _dict(read_json(job / "prepared.json", {}))
        audio = _number(prepared.get("duration"))
        source = _number(prepared.get("source_duration"))
    if audio is None or source is None:
        return None
    return math.inf if audio + BOUNDED_TOLERANCE >= source else round(audio)


def _existing_source(config: AppConfig, old: str, meta: dict) -> Path | None:
    name = meta.get("source_name")
    size = _dict(meta.get("identity")).get("size")
    candidates = [meta.get("archived_source_path"), _dict(meta.get("identity")).get("source")]
    paths = [Path(c) for c in candidates if isinstance(c, str) and c]
    if isinstance(name, str):
        paths.append(config.processed_dir / old / name)
        paths.append(config.input_dir / name)
    for path in paths:
        try:
            if path.is_file() and (not isinstance(size, int) or path.stat().st_size == size):
                return path
        except OSError:
            continue
    return None


def plan_migration(config: AppConfig) -> list[PlanItem]:
    roots = _roots(config)
    try:
        entries = sorted(p for p in config.process_dir.iterdir() if p.is_dir())
    except OSError:
        return []
    items: list[PlanItem] = []
    jobs: list[tuple[Path, dict]] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        meta = read_json(entry / "metadata.json")
        if not isinstance(meta, dict):
            migrated = any(
                parse_version_folder(child.name) and (child / "metadata.json").is_file()
                for child in entry.iterdir()
                if child.is_dir()
            )
            items.append(
                PlanItem(
                    entry.name,
                    "already-migrated" if migrated else "skipped",
                    "already a source folder" if migrated else "skipped (not a job)",
                )
            )
            continue
        jobs.append((entry, meta))

    # Pass 1: content keys wherever the media still exists, shared by (name, size).
    content_keys: dict[tuple, str] = {}
    located: dict[str, str] = {}
    for entry, meta in jobs:
        path = _existing_source(config, entry.name, meta)
        if path is None:
            continue
        try:
            key = source_key(path)
        except OSError:
            continue
        identity_key = (meta.get("source_name"), path.stat().st_size)
        content_keys[identity_key] = key
        located[entry.name] = key

    taken: set[tuple[str, str]] = set()
    processed_dirs: set[str] = set()
    for entry, meta in jobs:
        old = entry.name
        fingerprint = meta.get("config_fingerprint")
        name = meta.get("source_name")
        if not isinstance(fingerprint, str) or len(fingerprint) < 8 or not isinstance(name, str):
            items.append(PlanItem(old, "skipped", "metadata has no fingerprint or source name"))
            continue
        created = _parse_time(meta.get("created_at")) or _parse_time(meta.get("updated_at"))
        if created is None:
            items.append(PlanItem(old, "skipped", "metadata has no usable timestamp"))
            continue
        size = _dict(meta.get("identity")).get("size")
        size = (
            size if isinstance(size, int) else _dict(meta.get("source_metadata")).get("size_bytes")
        )
        size = size if isinstance(size, int) else 0
        key = located.get(old) or content_keys.get((name, size))
        basis = "content"
        if key is None:
            key, basis = metadata_source_key(size, name), "metadata"
        folder = find_source_folder([roots["process"], roots["output"]], key) or (
            source_folder_name(name, key)
        )
        seconds = _scope_seconds(meta, entry)
        max_duration = (
            None if seconds == math.inf else (float("nan") if seconds is None else float(seconds))
        )
        model = _model(meta, config.output_dir / old)
        version = version_folder_name(created, model, max_duration, fingerprint)
        item = PlanItem(old, "move", "", folder, version, basis)
        item.moves.append(Move("process", old, f"{folder}/{version}"))
        if (config.output_dir / old).is_dir():
            item.moves.append(Move("output", old, f"{folder}/{version}"))
        archived = config.processed_dir / old
        if archived.is_dir():
            target = config.processed_dir / folder
            if not target.exists() and folder not in processed_dirs:
                item.moves.append(Move("processed", old, folder))
            else:
                item.moves.extend(
                    Move("processed", f"{old}/{f.name}", f"{folder}/{f.name}", "file")
                    for f in sorted(archived.iterdir())
                    if f.is_file()
                )
        for move in item.moves:
            target_path = roots[move.root] / move.target
            if target_path.exists() or (move.root, move.target) in taken:
                item.status = "conflict"
                item.reason = f"target exists, refusing to overwrite: {move.root}/{move.target}"
                break
        if item.status == "move":
            taken.update((m.root, m.target) for m in item.moves)
            if any(m.root == "processed" for m in item.moves):
                processed_dirs.add(folder)
            if basis == "metadata":
                item.reason = "source file not found; key derived from recorded size and name"
            if not math.isfinite(max_duration if max_duration is not None else 0.0):
                item.reason = (item.reason + "; " if item.reason else "") + "scope unknown"
        items.append(item)
    return items


def format_plan(items: list[PlanItem]) -> list[str]:
    header = ("OLD JOB FOLDER", "ACTION", "KEY BASIS", "NEW LOCATION / REASON")
    rows = [header]
    for item in items:
        if item.status == "move":
            action, where = "move", f"{item.source_folder}/{item.version_folder}"
            if item.reason:
                where += f"  [{item.reason}]"
        else:
            action, where = item.status, item.reason
        rows.append((item.old, action, item.key_basis or "-", where))
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = []
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(3)) + "  " + row[3])
    return lines


def apply_migration(config: AppConfig, items: list[PlanItem], *, now: datetime | None = None):
    """Perform the planned renames; return ``(log_path, summary)``."""
    roots = _roots(config)
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    log_path = config.process_dir / f"layout-migration-{stamp}.json"
    log = {"schema_version": 1, "started_utc": datetime.now(UTC).isoformat(), "moves": []}
    summary = {"migrated": 0, "failed": 0, "refused": 0}
    for item in items:
        if item.status != "move":
            if item.status == "conflict":
                summary["refused"] += 1
            continue
        done: list[Move] = []
        failure = None
        for move in item.moves:
            source = roots[move.root] / move.source
            target = roots[move.root] / move.target
            try:
                if target.exists():
                    raise FileExistsError(f"target exists: {move.root}/{move.target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, target)
            except OSError as exc:
                failure = f"{move.root}/{move.source}: {exc}"
                break
            done.append(move)
            log["moves"].append(
                {
                    "job": item.old,
                    "root": move.root,
                    "from": move.source,
                    "to": move.target,
                    "kind": move.kind,
                    "result": "moved",
                }
            )
        if failure is not None:
            for move in reversed(done):
                try:
                    os.rename(roots[move.root] / move.target, roots[move.root] / move.source)
                    result = "rolled-back"
                except OSError as exc:
                    result = f"rollback-failed: {exc}"
                log["moves"].append(
                    {
                        "job": item.old,
                        "root": move.root,
                        "from": move.target,
                        "to": move.source,
                        "kind": move.kind,
                        "result": result,
                    }
                )
            log["moves"].append({"job": item.old, "result": f"failed: {failure}"})
            summary["failed"] += 1
        else:
            summary["migrated"] += 1
            if item.key_basis == "metadata":
                marker = config.process_dir / item.source_folder / KEY_BASIS_NAME
                if not marker.exists():
                    atomic_json(marker, {"source_key_basis": "metadata"})
        log["updated_utc"] = datetime.now(UTC).isoformat()
        atomic_json(log_path, log)
    log["updated_utc"] = datetime.now(UTC).isoformat()
    log["summary"] = summary
    atomic_json(log_path, log)
    summary["catalog"] = rebuild_all(config)
    return log_path, summary
