"""Shared synthetic fixtures: no media, models or network."""

from pathlib import Path

from audio_transcript.config import AppConfig
from audio_transcript.domain.models import (
    AudioChunk,
    MediaInfo,
    PreparedAudio,
    Segment,
    TranscriptionError,
)


def make_config(root: Path, **overrides) -> AppConfig:
    options = {
        "input_dir": root / "input",
        "processed_dir": root / "processed",
        "process_dir": root / "process",
        "output_dir": root / "output",
        "backend": "mock",
        "model": "mock",
        "vad": "none",
    }
    options.update(overrides)
    return AppConfig(**options)


class FakeProcessor:
    """Returns one-second chunks; ``trailing_silence`` extends only the analysed duration."""

    def __init__(self, chunk_count=2, *, trailing_silence=0.0, chunk_seconds=1.0, empty=False):
        self.chunk_count = chunk_count
        self.trailing_silence = trailing_silence
        self.chunk_seconds = chunk_seconds
        self.empty = empty
        self.prepare_calls = 0
        self.prepare_kwargs: list[dict] = []
        self.split_calls: list[AudioChunk] = []
        self.split_error: str | None = None

    @property
    def audio_duration(self):
        return self.chunk_count * self.chunk_seconds + self.trailing_silence

    def probe(self, source):
        return MediaInfo(duration=self.audio_duration, sample_rate=16000, channels=1, codec="fake")

    def prepare(self, source, destination, *, sample_rate, segmentation, max_duration=None):
        self.prepare_calls += 1
        self.prepare_kwargs.append(
            {"segmentation": segmentation, "sample_rate": sample_rate, "max_duration": max_duration}
        )
        chunks = []
        count = 0 if self.empty else self.chunk_count
        for index in range(count):
            path = destination / f"chunk_{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic test fixture")
            start = index * self.chunk_seconds
            chunks.append(AudioChunk(path, start, start + self.chunk_seconds, index))
        return PreparedAudio(
            chunks=tuple(chunks),
            duration=self.audio_duration,
            speech_seconds=count * self.chunk_seconds,
            segmentation={"detector": segmentation.detector, "synthetic": True},
        )

    def split(self, chunk, destination):
        self.split_calls.append(chunk)
        if self.split_error is not None:
            raise TranscriptionError(self.split_error)
        destination.mkdir(parents=True, exist_ok=True)
        middle = (chunk.start + chunk.end) / 2
        parts = []
        for label, start, end in (("a", chunk.start, middle), ("b", middle, chunk.end)):
            path = destination / f"chunk_{chunk.index}{label}.wav"
            path.write_bytes(b"synthetic split fixture")
            parts.append(AudioChunk(path, start, end, chunk.index, label))
        return parts


class FakeBackend:
    """Scriptable backend; ``behavior(chunk, temperature)`` may return segments or raise."""

    name = "mock"

    def __init__(self, failures=(), behavior=None):
        self.failures = set(failures)
        self.behavior = behavior
        self.calls = []
        self.attempts = []

    def check(self):
        return None

    def transcribe(self, chunk, *, language, temperature=None, use_prompt=True):
        self.calls.append(chunk.index)
        self.attempts.append((chunk.index, chunk.part, temperature))
        self.prompt_flags = getattr(self, "prompt_flags", []) + [use_prompt]
        if chunk.index in self.failures:
            raise RuntimeError("synthetic backend failure")
        if self.behavior is not None:
            return self.behavior(chunk, temperature)
        return [Segment(chunk.start, chunk.end, f"chunk {chunk.index}{chunk.part}")]

    def close(self):
        return None
