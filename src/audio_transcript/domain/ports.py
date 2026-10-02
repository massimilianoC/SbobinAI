"""Interfaces implemented by infrastructure adapters."""

from pathlib import Path
from typing import Protocol

from .models import AudioChunk, MediaInfo, Segment, Transcript


class MediaProcessor(Protocol):
    def probe(self, source: Path) -> MediaInfo: ...

    def prepare(
        self,
        source: Path,
        destination: Path,
        *,
        chunk_seconds: float,
        sample_rate: int,
        max_duration: float | None = None,
    ) -> list[AudioChunk]: ...


class TranscriptionBackend(Protocol):
    name: str

    def check(self) -> None: ...

    def transcribe(self, chunk: AudioChunk, *, language: str | None) -> list[Segment]: ...

    def close(self) -> None: ...


class TranscriptExporter(Protocol):
    def export(self, transcript: Transcript, destination: Path) -> None: ...
