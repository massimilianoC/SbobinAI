"""Lazy adapter for the historical Nexa Qwen2-Audio inference interface."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from audio_transcript.domain.models import AudioChunk, Segment, TranscriptionError


class NexaBackend:
    """Use Nexa's legacy ``NexaAudioLMInference`` class when explicitly installed."""

    name = "nexa"
    _module_name = "nexa.gguf.nexa_inference_audio_lm"

    def __init__(
        self,
        model: str = "qwen2audio",
        device: str = "auto",
        local_path: Path | str | None = None,
        projector_local_path: Path | str | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.local_path = Path(local_path) if local_path is not None else None
        self.projector_local_path = (
            Path(projector_local_path) if projector_local_path is not None else None
        )
        self._inference = None

    def check(self) -> None:
        """Validate runtime compatibility without initializing or downloading a model."""
        module = self._import_runtime()
        inference_type = getattr(module, "NexaAudioLMInference", None)
        if inference_type is None or not callable(inference_type):
            raise TranscriptionError(
                "The installed Nexa runtime does not expose the verified legacy "
                "NexaAudioLMInference API. Install a compatible Nexa SDK revision."
            )
        if not callable(getattr(inference_type, "inference", None)):
            # Instance methods are normally defined on the class and are callable there.
            raise TranscriptionError(
                "The installed NexaAudioLMInference class has no compatible inference(audio_path, prompt) method."
            )
        if not any(callable(getattr(inference_type, name, None)) for name in ("cleanup", "close")):
            raise TranscriptionError(
                "The installed NexaAudioLMInference class exposes no cleanup() or close() method."
            )
        if self.local_path is None or self.projector_local_path is None:
            raise TranscriptionError(
                "Automatic Nexa model downloads are disabled. Configure both local model and "
                "projector files (--model-path and --projector-path) before inference."
            )
        for label, path in (
            ("model", self.local_path),
            ("projector", self.projector_local_path),
        ):
            if not path.is_file():
                raise TranscriptionError(
                    f"The configured local Nexa {label} file does not exist: {path}"
                )

    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None,
        temperature: float | None = None,
        use_prompt: bool = True,
    ) -> list[Segment]:
        # The legacy Nexa inference API exposes no sampling control; the value is ignored.
        del temperature
        if self._inference is None:
            self.check()
            module = self._import_runtime()
            inference_type = module.NexaAudioLMInference
            try:
                self._inference = inference_type(
                    model_path=self.model,
                    local_path=str(self.local_path),
                    projector_local_path=str(self.projector_local_path),
                    device=self.device,
                )
            except Exception as exc:
                raise TranscriptionError(
                    "Could not initialize Nexa Qwen2-Audio. Verify the legacy runtime and "
                    "configured local model and projector files."
                ) from exc
            except SystemExit as exc:
                raise TranscriptionError(
                    "Nexa rejected the configured local model or projector files. "
                    "Check that they match the selected Qwen2-Audio model."
                ) from exc

        language_hint = f" The expected language is {language}." if language else ""
        prompt = (
            "Transcribe all audible speech faithfully in its original language. Preserve the "
            "spoken wording. Do not translate, summarize, complete inaudible words, or invent "
            "speaker identities. Return only the transcript text."
            f"{language_hint}"
        )
        inference_error = None
        try:
            response = self._inference.inference(str(Path(chunk.path)), prompt=prompt)
        except Exception as exc:
            inference_error = exc
        finally:
            cleanup = getattr(self._inference, "cleanup", None) or getattr(
                self._inference, "close", None
            )
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    if inference_error is None:
                        raise TranscriptionError(
                            f"Could not release Nexa resources after audio chunk {chunk.index}."
                        ) from exc
        if inference_error is not None:
            raise TranscriptionError(
                f"Nexa inference failed for audio chunk {chunk.index}: {inference_error}"
            ) from inference_error
        if not isinstance(response, str) or not response.strip():
            raise TranscriptionError(
                f"Nexa returned an empty transcript for audio chunk {chunk.index}; "
                "the chunk may contain silence or inference may have failed."
            )
        return [
            Segment(
                start=chunk.start,
                end=chunk.end,
                text=response.strip(),
                speaker=None,
                timing_source="chunk",
            )
        ]

    def close(self) -> None:
        inference, self._inference = self._inference, None
        if inference is None:
            return
        close = getattr(inference, "close", None) or getattr(inference, "cleanup", None)
        if callable(close):
            close()

    @classmethod
    def _import_runtime(cls) -> ModuleType:
        try:
            module = importlib.import_module(cls._module_name)
        except (ImportError, OSError, SystemExit) as exc:
            raise TranscriptionError(
                "Nexa's legacy Qwen2-Audio runtime is unavailable or incompatible. "
                "Install a Nexa SDK build that provides "
                "nexa.gguf.nexa_inference_audio_lm.NexaAudioLMInference."
            ) from exc
        return module
