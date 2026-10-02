"""Shared data models for media processing and transcription."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start: float
    end: float
    index: int


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    timing_source: str = "chunk"


@dataclass
class Transcript:
    job_id: str
    backend: str
    model: str
    language: str | None
    duration: float
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = "1.0"
    source_metadata: dict = field(default_factory=dict)
    audio_metadata: dict = field(default_factory=dict)


class TranscriptionError(Exception):
    """An actionable failure in configuration or processing."""
