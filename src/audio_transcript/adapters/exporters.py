"""Validated UTF-8 transcript artifacts with explicit timing provenance."""

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from audio_transcript.domain.models import Transcript, TranscriptionError


def _atomic_text(path: Path, text: str) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _timestamp(seconds: float, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    seconds_total, millis = divmod(milliseconds, 1000)
    minutes_total, second = divmod(seconds_total, 60)
    hour, minute = divmod(minutes_total, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}{separator}{millis:03d}"


class FileExporter:
    """Export a domain transcript without inventing alignment or speakers."""

    def export(self, transcript: Transcript, destination: Path) -> None:
        if not math.isfinite(transcript.duration) or transcript.duration <= 0:
            raise TranscriptionError("Transcript duration must be finite and positive.")
        previous_end = 0.0
        for segment in transcript.segments:
            if (
                not math.isfinite(segment.start)
                or not math.isfinite(segment.end)
                or segment.start < previous_end - 0.000001
                or segment.start < 0
                or segment.end <= segment.start
                or segment.end > transcript.duration + 0.000001
                or round(segment.end * 1000) <= round(segment.start * 1000)
            ):
                raise TranscriptionError(
                    "Transcript segments require ordered times within duration."
                )
            if not isinstance(segment.text, str) or not segment.text.strip():
                raise TranscriptionError("Transcript segment text must be a nonempty string.")
            if segment.speaker is not None and not isinstance(segment.speaker, str):
                raise TranscriptionError("Speaker labels must be strings or null.")
            previous_end = segment.end

        payload = asdict(transcript)
        warnings = list(transcript.warnings)
        if any(s.timing_source == "chunk" for s in transcript.segments):
            warnings.append(
                "Subtitle timestamps are coarse audio chunk boundaries, not speech alignment."
            )
        if transcript.backend == "mock":
            warnings.append("Synthetic mock output; no speech recognition was performed.")
        payload["warnings"] = list(dict.fromkeys(warnings))
        plain = "\n".join(s.text for s in transcript.segments)
        markdown = [
            "# Transcript",
            "",
            f"Backend: {transcript.backend}",
            f"Model: {transcript.model}",
            "",
        ]
        markdown.extend(f"> {warning}" for warning in payload["warnings"])
        markdown.append("")
        srt: list[str] = []
        vtt: list[str] = ["WEBVTT", ""]
        for index, segment in enumerate(transcript.segments, 1):
            # Cue blocks cannot contain blank lines; normalize these only in subtitles.
            text = "\n".join(line.strip() for line in segment.text.splitlines() if line.strip())
            if segment.speaker:
                speaker = " ".join(segment.speaker.split())
                text = f"[{speaker}] {text}"
            markdown.extend(
                [
                    f"## {_timestamp(segment.start, '.')} – {_timestamp(segment.end, '.')}",
                    "",
                    segment.text,
                    "",
                ]
            )
            srt.extend(
                [
                    str(index),
                    f"{_timestamp(segment.start, ',')} --> {_timestamp(segment.end, ',')}",
                    text,
                    "",
                ]
            )
            # Escape WebVTT markup characters to preserve literal transcript text.
            vtt_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            vtt.extend(
                [
                    str(index),
                    f"{_timestamp(segment.start, '.')} --> {_timestamp(segment.end, '.')}",
                    vtt_text,
                    "",
                ]
            )
        report = {
            "schema_version": transcript.schema_version,
            "job_id": transcript.job_id,
            "status": "mock_completed" if transcript.backend == "mock" else "completed",
            "backend": transcript.backend,
            "model": transcript.model,
            "duration_seconds": transcript.duration,
            "segment_count": len(transcript.segments),
            "warnings": payload["warnings"],
            "source_metadata": transcript.source_metadata,
            "audio_metadata": transcript.audio_metadata,
        }
        destination.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "transcript.json": json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            "transcript.txt": plain + ("\n" if plain else ""),
            "transcript.md": "\n".join(markdown),
            "transcript.srt": "\n".join(srt) + ("\n" if srt else ""),
            "transcript.vtt": "\n".join(vtt) + "\n",
            "report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        }
        for filename, content in artifacts.items():
            _atomic_text(destination / filename, content)
