"""Explicit diagnostic backend for pipeline wiring and non-inference tests."""

from audio_transcript.domain.models import AudioChunk, Segment


class MockBackend:
    name = "mock"
    diagnostic_text = "MOCK TRANSCRIPT — synthetic output; no speech recognition performed."

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def check(self) -> None:
        return None

    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None,
        temperature: float | None = None,
        use_prompt: bool = True,
    ) -> list[Segment]:
        del language, temperature, use_prompt
        return [
            Segment(
                start=chunk.start,
                end=chunk.end,
                text=self.diagnostic_text,
                speaker=None,
                timing_source="chunk",
            )
        ]

    def close(self) -> None:
        return None
