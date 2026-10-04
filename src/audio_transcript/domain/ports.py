"""Interfaces implemented by infrastructure adapters."""

from pathlib import Path
from typing import Protocol

from .models import (
    AudioChunk,
    MediaInfo,
    PreparedAudio,
    Segment,
    SegmentationSettings,
    Transcript,
    VoiceActivity,
)


class SpeechDetector(Protocol):
    name: str

    def check(self) -> None:
        """Raise TranscriptionError with an actionable message if unusable."""
        ...

    def detect(self, wav_path: Path) -> VoiceActivity:
        """Score a mono 16-bit PCM WAV file frame by frame."""
        ...


class MediaProcessor(Protocol):
    def probe(self, source: Path) -> MediaInfo: ...

    def prepare(
        self,
        source: Path,
        destination: Path,
        *,
        sample_rate: int,
        segmentation: SegmentationSettings,
        max_duration: float | None = None,
    ) -> PreparedAudio:
        """Normalize the first audio stream and plan speech chunks.

        Every returned chunk lasts at most ``segmentation.max_chunk_seconds``;
        chunks are ordered, non-overlapping, indexed from 0 and stored inside
        ``destination``.
        """
        ...

    def export_audio(self, source: Path, destination: Path, *, sample_rate: int) -> MediaInfo:
        """Write the first audio stream as lossless mono FLAC at ``sample_rate``.

        This is the audio the model hears, kept so a recording can be transcribed
        again after its original file is deleted. Returns the probe of the new file.
        """
        ...

    def split(self, chunk: AudioChunk, destination: Path) -> list[AudioChunk]:
        """Split one prepared chunk at its quietest interior pause.

        Returns at least two sub-chunks with the parent ``index``, distinct
        ``part`` suffixes and source-relative times that exactly cover the
        parent span. Raise TranscriptionError when the chunk is too short.
        """
        ...


class TranscriptionBackend(Protocol):
    name: str

    def check(self) -> None: ...

    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None,
        temperature: float | None = None,
        use_prompt: bool = True,
    ) -> list[Segment]:
        """Transcribe one chunk.

        ``temperature`` overrides the configured value for a fallback attempt.
        ``use_prompt=False`` drops the configured prompt/context for this call
        (built-in instructions still apply); the pipeline uses it only after the
        model echoed that context.
        An empty list means the model explicitly reported no speech. Raise
        TransientBackendError or DegenerateOutputError where applicable.
        """
        ...

    def close(self) -> None: ...


class TranscriptExporter(Protocol):
    def export(self, transcript: Transcript, destination: Path) -> None: ...
