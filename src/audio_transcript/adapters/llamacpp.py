"""Local llama.cpp server adapter using its OpenAI-compatible HTTP API."""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from audio_transcript.domain.models import AudioChunk, Segment, TranscriptionError


class LlamaCppBackend:
    """Transcribe WAV chunks through a local llama.cpp multimodal server."""

    name = "llamacpp"

    def __init__(
        self,
        model: str = "qwen2-audio-7b",
        base_url: str = "http://127.0.0.1:8088",
        timeout: float = 180,
        max_tokens: int = 1024,
        prompt: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        response_mode: str = "json",
    ) -> None:
        self.model = model
        self.base_url = _normalize_base_url(base_url)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a finite number greater than zero")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite number greater than zero")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("prompt must be a non-empty string or None")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number between 0 and 2")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be a finite number between 0 and 2")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
            raise ValueError("seed must be an integer between 0 and 4294967295")
        if not isinstance(response_mode, str) or response_mode not in {"json", "plain"}:
            raise ValueError("response_mode must be 'json' or 'plain'")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.temperature = float(temperature)
        self.seed = seed
        self.response_mode = response_mode
        self._server_model_id: str | None = None

    def check(self) -> None:
        """Check server readiness and that the requested model is loaded."""
        health = self._get_json("/health", purpose="check llama.cpp server health")
        if health.get("status") != "ok":
            raise TranscriptionError(
                "The llama.cpp server is reachable but not ready. Wait for model loading to "
                "finish, then rerun doctor."
            )

        models = self._get_json("/v1/models", purpose="list llama.cpp models")
        entries = models.get("data")
        if not isinstance(entries, list):
            raise TranscriptionError("llama.cpp returned an invalid /v1/models response.")
        valid_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]
        if not valid_entries:
            raise TranscriptionError(
                "llama.cpp reports no loaded model. Start llama-server with the desired model first."
            )

        selected = next((entry for entry in valid_entries if entry["id"] == self.model), None)
        if selected is None:
            loaded = ", ".join(entry["id"] for entry in valid_entries)
            raise TranscriptionError(
                f"Model '{self.model}' is not loaded in llama.cpp. Loaded model ids: {loaded}. "
                "Start the server with --alias matching the configured model or update the model setting."
            )
        _require_audio_capability_if_reported(selected)
        self._server_model_id = selected["id"]

    def transcribe(self, chunk: AudioChunk, *, language: str | None) -> list[Segment]:
        if self._server_model_id is None:
            self.check()
        try:
            audio_data = Path(chunk.path).read_bytes()
        except OSError as exc:
            raise TranscriptionError(f"Could not read audio chunk {chunk.index}: {exc}") from exc
        if not audio_data:
            raise TranscriptionError(f"Audio chunk {chunk.index} is empty.")

        prompt = self._build_prompt(language)
        payload = {
            "model": self._server_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_data).decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stream": False,
            "temperature": self.temperature,
        }
        if self.response_mode == "json":
            payload["response_format"] = _transcript_json_schema()
        response = self._post_json(
            "/v1/chat/completions",
            payload,
            purpose=f"transcribe audio chunk {chunk.index}",
        )
        self._save_response_audit(chunk, prompt, response)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise TranscriptionError(
                f"llama.cpp returned no completion for audio chunk {chunk.index}."
            )
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise TranscriptionError(
                f"Transcription of audio chunk {chunk.index} reached max_tokens={self.max_tokens}. "
                "Increase the configured token limit or use shorter audio chunks; the transcript "
                "may be incomplete."
            )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if self.response_mode == "json":
            text = _extract_json_transcript(content, chunk.index)
        else:
            text = _extract_text(content)
        if not text:
            raise TranscriptionError(
                f"llama.cpp returned an empty transcript for audio chunk {chunk.index}. "
                "Confirm that the loaded model and projector support audio input."
            )
        return [
            Segment(
                start=chunk.start,
                end=chunk.end,
                text=text,
                speaker=None,
                timing_source="chunk",
            )
        ]

    def effective_prompt(self, language: str | None) -> str:
        """Return the exact instruction submitted for output provenance."""
        return self._build_prompt(language)

    def _build_prompt(self, language: str | None) -> str:
        if self.prompt is not None:
            prompt = self.prompt.strip()
        elif self.response_mode == "json":
            prompt = (
                "Trascrivi letteralmente il parlato nell'audio nella lingua originale. "
                "Nel campo transcript scrivi soltanto le parole pronunciate, in ordine. "
                "Nessuna introduzione, commento, traduzione o nome inventato."
            )
        else:
            prompt = (
                "Trascrivi letteralmente il parlato nell'audio nella lingua originale. "
                "Restituisci soltanto le parole pronunciate, in ordine, senza introduzioni, "
                "commenti, traduzioni o nomi inventati."
            )
        if language:
            prompt += (
                f"\nLingua indicata: {language}. Mantieni comunque la lingua originale "
                "effettivamente parlata."
            )
        return prompt

    def _save_response_audit(self, chunk: AudioChunk, prompt: str, response: dict) -> None:
        response_dir = Path(chunk.path).parent / "responses"
        destination = response_dir / f"{chunk.index:05d}.json"
        audit = {
            "schema_version": 1,
            "backend": self.name,
            "model": self._server_model_id,
            "chunk": {"index": chunk.index, "start": chunk.start, "end": chunk.end},
            "request": {
                "prompt": prompt,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
                "response_mode": self.response_mode,
                "response_format": (
                    _transcript_json_schema() if self.response_mode == "json" else None
                ),
            },
            "response": response,
        }
        temporary_path = None
        try:
            response_dir.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=response_dir
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(audit, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary_path, destination)
        except Exception as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise TranscriptionError(
                f"Could not save raw response audit for audio chunk {chunk.index}: {exc}"
            ) from exc

    def close(self) -> None:
        """The local llama.cpp server is managed outside this process."""

    def _get_json(self, path: str, *, purpose: str) -> dict:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        return self._send_json(request, purpose=purpose)

    def _post_json(self, path: str, payload: dict, *, purpose: str) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        return self._send_json(request, purpose=purpose)

    def _send_json(self, request: urllib.request.Request, *, purpose: str) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = _error_detail(exc.read())
            finally:
                exc.close()
            raise TranscriptionError(
                f"Could not {purpose}: llama.cpp returned HTTP {exc.code}"
                f"{': ' + detail if detail else ''}. Check that the server is running with an "
                "audio-capable model and projector."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise TranscriptionError(
                f"Could not {purpose} at {request.full_url}: {reason}. "
                "Start llama-server and verify its host and port."
            ) from exc
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TranscriptionError(
                f"llama.cpp returned invalid JSON while trying to {purpose}."
            ) from exc
        if not isinstance(parsed, dict):
            raise TranscriptionError(
                f"llama.cpp returned an invalid response while trying to {purpose}."
            )
        if isinstance(parsed.get("error"), dict):
            detail = _error_detail(json.dumps(parsed["error"], ensure_ascii=False).encode("utf-8"))
            raise TranscriptionError(
                f"Could not {purpose}: {detail or 'server reported an error'}."
            )
        return parsed


def _normalize_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("Credentials in base_url are not supported")
    host = parsed.hostname.lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("llama.cpp backend only accepts a local loopback server URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _require_audio_capability_if_reported(model: dict) -> None:
    metadata = model.get("meta")
    architecture = metadata.get("architecture") if isinstance(metadata, dict) else None
    modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
    if isinstance(modalities, list) and "audio" not in modalities:
        raise TranscriptionError(
            "The loaded llama.cpp model metadata does not advertise audio input. "
            "Load an audio-capable model with its matching projector."
        )
    if model.get("multimodal") is False:
        raise TranscriptionError(
            "The loaded llama.cpp model is not marked as multimodal. "
            "Load an audio-capable model with its matching projector."
        )


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type", "text") == "text"
        ]
        return "".join(part for part in parts if isinstance(part, str)).strip()
    return ""


def _transcript_json_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "transcription",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"transcript": {"type": "string"}},
                "required": ["transcript"],
                "additionalProperties": False,
            },
        },
    }


def _extract_json_transcript(content: object, chunk_index: int) -> str:
    if not isinstance(content, str):
        raise TranscriptionError(
            f"llama.cpp returned non-text JSON content for audio chunk {chunk_index}."
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TranscriptionError(
            f"llama.cpp returned invalid JSON for audio chunk {chunk_index}; "
            "the constrained transcript response could not be parsed."
        ) from exc
    if not isinstance(parsed, dict) or set(parsed) != {"transcript"}:
        raise TranscriptionError(
            f"llama.cpp returned an invalid transcript JSON object for audio chunk {chunk_index}."
        )
    transcript = parsed["transcript"]
    if not isinstance(transcript, str):
        raise TranscriptionError(
            f"llama.cpp returned a non-string transcript for audio chunk {chunk_index}."
        )
    return transcript.strip()


def _error_detail(body: bytes) -> str:
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace").strip()[-500:]
    if isinstance(parsed, dict):
        error = parsed.get("error", parsed)
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
            if isinstance(message, str):
                return message.strip()[-500:]
        if isinstance(error, str):
            return error.strip()[-500:]
    return ""
