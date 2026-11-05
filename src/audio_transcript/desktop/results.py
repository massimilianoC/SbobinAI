"""Facts about a finished version for the results screen, read from disk, never assumed.

Marks are deliberately narrow: only audio, video and ZIP files are *verified* (re-probed or
read back now); text and report files are *complete* (present, non-empty, equal in size to
their canonical copy, and parseable when JSON). Nothing here claims transcript correctness.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from ..application.delivery import ArtifactDelivery
from ..application.state import read_json

KINDS = {
    "transcript.txt": "Text",
    "transcript.md": "Text",
    "transcript.srt": "Subtitles",
    "transcript.vtt": "Subtitles",
    "transcript.json": "Data",
    "report.json": "Data",
    "run-report.md": "Report",
    "audio.mp3": "Audio",
    "audio.flac": "Audio",
    "video.mp4": "Video",
    "intermediate.zip": "Archive",
}
MAX_PARSE_BYTES = 64 * 1024 * 1024
MAX_ZIP_BYTES = 1024 * 1024 * 1024
MAX_FILES = 40


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _last_packet_end(processor, path: Path) -> float | None:
    """End time of the last audio packet, read from the container without decoding.

    A truncated file keeps the duration written in its header, so the header alone cannot
    prove the audio data is all there. Returns None when it cannot be determined.
    """
    ffprobe = getattr(processor, "ffprobe", None)
    if ffprobe is None:
        return None
    flags = (
        {"creationflags": subprocess.CREATE_NO_WINDOW}
        if hasattr(subprocess, "CREATE_NO_WINDOW")
        else {}
    )
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "packet=pts_time,duration_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120, check=False, **flags,
        )  # fmt: skip
        last = next((ln for ln in reversed(out.stdout.splitlines()) if ln.strip()), "")
        # Packets with side data end in an extra comma ("59.97,0.022,"), so split, not partition.
        fields = [part.strip() for part in last.split(",")]
        length = fields[1] if len(fields) > 1 else ""
        return float(fields[0]) + (float(length) if length not in {"", "N/A"} else 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _render_label(delivery) -> str:
    """Which encoder made the preview, from the delivery record ("" when unknown)."""
    record = (read_json(delivery.work / "delivery.json", {}) or {}).get("video_render")
    if not isinstance(record, dict) or not record.get("encoder"):
        return ""
    if record["encoder"] == "h264_nvenc":
        return " · NVENC" + (", GPU decode" if record.get("gpu_decode") else "")
    return " · CPU (x264)"


def _check_media(path: Path, processor, expected: float | None) -> tuple[str, str]:
    info = processor.probe(path)
    if path.name == "video.mp4":
        if not info.has_video or (info.video_height or 10**6) > 720 or info.codec != "aac":
            return "problem", "Not a 720p-or-smaller H.264/AAC video"
        detail = f"{info.video_height}p"
    else:
        wanted = path.suffix[1:]
        if info.codec != wanted:
            return "problem", f"Unexpected codec {info.codec}"
        detail = wanted.upper()
    if expected and abs(info.duration - expected) > max(1.0, expected * 0.001):
        return "problem", f"Duration {_duration(info.duration)} does not match the selection"
    end = _last_packet_end(processor, path)
    if end is None:
        return "verified", f"{detail} · {_duration(info.duration)} · data end not checked"
    if abs(end - info.duration) > max(1.0, info.duration * 0.001):
        return (
            "problem",
            f"Audio data ends at {_duration(end)} but the header says {_duration(info.duration)}",
        )
    return "verified", f"{detail} · {_duration(info.duration)}"


def _check_file(path: Path, canonical: Path, processor, expected, partial: bool) -> tuple[str, str]:
    size = path.stat().st_size
    if size == 0:
        return "problem", "The file is empty"
    if partial:
        return "partial", "Partial result of an unfinished job"
    if path.name in {"audio.mp3", "audio.flac", "video.mp4"}:
        return _check_media(path, processor, expected)
    if path.name == "intermediate.zip":
        if size > MAX_ZIP_BYTES:
            return "problem", "Too large to read back here"
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                return "problem", f"Corrupt entry {bad}"
            return "verified", f"{len(archive.namelist())} files"
    if canonical.is_file() and canonical.stat().st_size != size:
        return "problem", "Differs in size from the canonical copy"
    if path.suffix == ".json" and size <= MAX_PARSE_BYTES:
        json.loads(path.read_text(encoding="utf-8"))
    return "complete", "Present, non-empty" + (
        ", matches the canonical copy" if canonical.is_file() else ""
    )


def inspect_files(
    folder: Path, config, job_id: str | None, processor, *, partial: bool = False
) -> dict:
    """List a version folder with one honest state per file; never raises."""
    delivery = ArtifactDelivery(config, job_id) if job_id else None
    expected = None
    if delivery is not None:
        meta = read_json(delivery.existing("transcript.json"), {})
        value = (meta.get("audio_metadata") or {}).get("duration_seconds") if meta else None
        expected = float(value) if isinstance(value, (int, float)) else None
    rows = []
    try:
        entries = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    except OSError:
        entries = []
    for path in entries[:MAX_FILES]:
        canonical = delivery.canonical / path.name if delivery is not None else path
        try:
            size = path.stat().st_size
            state, note = _check_file(path, canonical, processor, expected, partial)
            if path.name == "video.mp4" and state == "verified" and delivery is not None:
                note += _render_label(delivery)
        except Exception as exc:  # an unreadable or corrupt file is a result, not an error
            state, note, size = "problem", str(exc)[:160] or type(exc).__name__, 0
        rows.append(
            {
                "name": path.name,
                "kind": KINDS.get(path.name, "Other"),
                "size": size,
                "state": state,
                "note": note,
            }
        )
    problems = sum(row["state"] == "problem" for row in rows)
    return {"files": rows, "problems": problems, "total": len(rows)}


def queued_file_outcome(config, job_id: str, queued: Path, status: str | None) -> dict:
    """What actually happened to the queued recording, from the job's own records."""
    meta = read_json(config.process_dir / job_id / "metadata.json", {}) or {}
    # A reused version keeps records from earlier runs, so the files on disk come first.
    try:
        present = queued.is_file()
    except OSError:
        present = False
    if not present:
        archived = meta.get("archived_source_path")
        if archived and Path(archived).is_file():
            return {"state": "archived", "text": "Archived in processed, as the queue policy says."}
        disposition = meta.get("source_disposition")
        if isinstance(disposition, dict) and disposition.get("original_deleted"):
            policy = disposition.get("policy")
            kept = " A lossless FLAC copy of its audio was kept." if policy == "keep-audio" else ""
            return {
                "state": "deleted",
                "text": f"Deleted from input after a complete transcription.{kept}",
            }
        return {"state": "gone", "text": "No longer in input."}
    reason = meta.get("disposition_warning") or meta.get("archive_warning")
    if reason:
        why = str(reason)
    elif status not in {"completed", "skipped"}:
        why = "the job did not finish"
    elif isinstance(meta.get("scope"), dict) and meta["scope"].get("kind") != "full":
        why = "only part of the recording was transcribed"
    elif meta.get("backend") == "mock":
        why = "mock output never archives or deletes a recording"
    elif queued.parent.resolve() != config.input_dir.resolve():
        why = "files in subfolders are not archived automatically"
    else:
        why = "archiving was not applied"
    return {"state": "kept", "text": f"Kept in input: {why}."}
