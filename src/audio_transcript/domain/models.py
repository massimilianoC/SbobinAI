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
    """A prepared WAV slice; start/end are seconds relative to the source audio.

    ``part`` is empty for planned chunks and a suffix such as ``"a"``/``"b"`` for
    sub-chunks produced by a fallback split of chunk ``index``.
    """

    path: Path
    start: float
    end: float
    index: int
    part: str = ""


@dataclass(frozen=True)
class SpeechRegion:
    """A source-relative time span in seconds."""

    start: float
    end: float


@dataclass(frozen=True)
class VoiceActivity:
    """Frame-level speech probabilities produced by a speech detector.

    ``probabilities[i]`` covers ``[i * frame_seconds, (i + 1) * frame_seconds)``.
    Values are in [0, 1]; ``threshold`` is the speech onset level the detector
    recommends. ``details`` holds JSON-serializable provenance such as an
    adaptive energy threshold.
    """

    frame_seconds: float
    probabilities: tuple[float, ...]
    threshold: float
    detector: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentationSettings:
    """Pause-aware chunk planning bounds. Durations are seconds.

    ``detector`` is ``"silero"`` (Silero VAD ONNX on CPU), ``"energy"`` (FFmpeg
    frame energy, no extra dependencies) or ``"none"`` (contiguous fixed chunks
    of ``max_chunk_seconds``, the historical behaviour).
    """

    detector: str = "silero"
    max_chunk_seconds: float = 15.0
    threshold: float = 0.5
    min_speech_seconds: float = 0.25
    min_silence_seconds: float = 0.1
    speech_pad_seconds: float = 0.2
    max_merge_gap_seconds: float = 1.0
    energy_margin_db: float = 15.0


@dataclass(frozen=True)
class PreparedAudio:
    """Result of media preparation.

    ``chunks`` contains only audio to transcribe and may be empty when no speech
    was detected. ``duration`` is the full normalized audio duration analysed,
    independent of where speech occurs. ``segmentation`` records the effective
    detector, settings and derived values for job provenance; it must be
    JSON-serializable.
    """

    chunks: tuple[AudioChunk, ...]
    duration: float
    speech_seconds: float
    segmentation: dict = field(default_factory=dict)
    # Optional processor stage timings in seconds (audio_extraction, speech_detection, ...).
    timings: dict = field(default_factory=dict)


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


class TransientBackendError(TranscriptionError):
    """The backend did not answer (connection, timeout, HTTP 5xx); an identical retry may succeed."""


class DegenerateOutputError(TranscriptionError):
    """The model answered but the output is unusable: token limit, repetition loop,
    prompt echo or invalid structure. An identical request will fail again, so the
    pipeline applies its fallback ladder instead of retrying unchanged."""
