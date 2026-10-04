"""Local llama.cpp server adapter using its OpenAI-compatible HTTP API."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

from audio_transcript.domain.models import (
    AudioChunk,
    DegenerateOutputError,
    Segment,
    TranscriptionError,
    TransientBackendError,
)
from audio_transcript.fsutil import replace_with_retry

RESPONSE_MODES = ("json", "plain", "qwen3-asr")
_PROMPT_ECHO_WORDS = 6
_AUDIT_LOCK = threading.Lock()  # serialises unique audit file name selection
_MIN_COMPRESSION_CHARS = 40
_QWEN3_ASR = re.compile(
    r"\s*language\s+(?P<language>[^<]*?)\s*<asr_text>(?P<text>.*)", re.DOTALL | re.IGNORECASE
)
_QWEN3_ASR_NONE = re.compile(r"\s*language\s+none\s*(?:<asr_text>\s*)?", re.IGNORECASE)
# Languages documented for Qwen3-ASR, keyed by ISO 639 code, as named in its
# ``language <Name><asr_text>`` output protocol.
QWEN3_ASR_LANGUAGES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


def qwen3_asr_language(language: str | None) -> str | None:
    """Map an ISO code or English name to the Qwen3-ASR language name, if supported."""
    if not language:
        return None
    key = language.strip().casefold()
    if key in QWEN3_ASR_LANGUAGES:
        return QWEN3_ASR_LANGUAGES[key]
    by_name = {name.casefold(): name for name in QWEN3_ASR_LANGUAGES.values()}
    return by_name.get(key)


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
        min_tokens: int = 48,
        tokens_per_second: float = 8.0,
        compression_ratio_threshold: float = 2.4,
        repeat_penalty: float = 1.0,
        dry_multiplier: float = 0.0,
        force_language: bool = True,
        collect_logprobs: bool = True,
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
        if not isinstance(response_mode, str) or response_mode not in RESPONSE_MODES:
            raise ValueError("response_mode must be 'json', 'plain', or 'qwen3-asr'")
        if isinstance(min_tokens, bool) or not isinstance(min_tokens, int) or min_tokens <= 0:
            raise ValueError("min_tokens must be a positive integer")
        for label, value, low_exclusive in (
            ("tokens_per_second", tokens_per_second, False),
            ("compression_ratio_threshold", compression_ratio_threshold, True),
            ("repeat_penalty", repeat_penalty, True),
            ("dry_multiplier", dry_multiplier, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (low_exclusive and value == 0)
            ):
                raise ValueError(f"{label} must be a finite number in its valid range")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.temperature = float(temperature)
        self.seed = seed
        self.response_mode = response_mode
        self.min_tokens = min_tokens
        self.tokens_per_second = float(tokens_per_second)
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.repeat_penalty = float(repeat_penalty)
        self.dry_multiplier = float(dry_multiplier)
        if not isinstance(force_language, bool):
            raise ValueError("force_language must be true or false")
        self.force_language = force_language
        if not isinstance(collect_logprobs, bool):
            raise ValueError("collect_logprobs must be true or false")
        self.collect_logprobs = collect_logprobs
        self._server_model_id: str | None = None
        self._check_lock = threading.Lock()
        # Per-thread: concurrent chunk requests must never read each other's metrics.
        self._local = threading.local()

    @property
    def _last_metrics(self) -> dict | None:
        return getattr(self._local, "metrics", None)

    @_last_metrics.setter
    def _last_metrics(self, value: dict | None) -> None:
        self._local.metrics = value

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

    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None,
        temperature: float | None = None,
        use_prompt: bool = True,
    ) -> list[Segment]:
        """Transcribe one chunk; an empty list means the model reported no speech."""
        self._last_metrics = None
        if self._server_model_id is None:
            with self._check_lock:
                if self._server_model_id is None:
                    self.check()
        effective_temperature = self._resolve_temperature(temperature)
        try:
            audio_data = Path(chunk.path).read_bytes()
        except OSError as exc:
            raise TranscriptionError(f"Could not read audio chunk {chunk.index}: {exc}") from exc
        if not audio_data:
            raise TranscriptionError(f"Audio chunk {chunk.index} is empty.")

        prompt = self.effective_prompt(language) if use_prompt else self.builtin_prompt(language)
        audio_part = {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(audio_data).decode("ascii"),
                "format": "wav",
            },
        }
        forced = self.forced_language(language)
        if self.response_mode == "qwen3-asr":
            messages: list[dict] = []
            if prompt:
                messages.append({"role": "system", "content": prompt})
            messages.append({"role": "user", "content": [audio_part]})
            if forced:
                # Assistant prefill with the model's own output prefix, as the official
                # Qwen3-ASR toolkit does for a fixed language; llama.cpp continues after it.
                messages.append({"role": "assistant", "content": _qwen3_asr_prefix(forced)})
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, audio_part]}]
        max_tokens = self.token_cap(chunk)
        payload = {
            "model": self._server_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "seed": self.seed,
            "stream": False,
            "temperature": effective_temperature,
        }
        if self.repeat_penalty != 1.0:
            payload["repeat_penalty"] = self.repeat_penalty
        if self.dry_multiplier != 0.0:
            payload["dry_multiplier"] = self.dry_multiplier
        if self.collect_logprobs:
            # Token log-probabilities feed the uncalibrated confidence proxy; never stored as text.
            payload["logprobs"] = True
        if self.response_mode == "json":
            payload["response_format"] = _transcript_json_schema()
        response = self._post_json(
            "/v1/chat/completions",
            payload,
            purpose=f"transcribe audio chunk {chunk.index}",
        )
        self._last_metrics = _response_metrics(response, max_tokens)
        outcome: dict = {}
        failure: DegenerateOutputError | None = None
        text = ""
        try:
            text = self._interpret(response, chunk, prompt, outcome, forced)
        except DegenerateOutputError as exc:
            failure = exc
        outcome["status"] = "degenerate" if failure else ("ok" if text else "no_speech")
        if failure is not None:
            outcome["error"] = str(failure)
        # Persist the raw response before any rejection so every attempt stays reviewable.
        self._save_response_audit(
            chunk,
            prompt,
            response,
            temperature=effective_temperature,
            max_tokens=max_tokens,
            outcome=outcome,
        )
        if failure is not None:
            raise failure
        if not text:
            return []
        return [
            Segment(
                start=chunk.start,
                end=chunk.end,
                text=text,
                speaker=None,
                timing_source="chunk",
            )
        ]

    def last_call_metrics(self) -> dict | None:
        """Numeric metadata of the latest call; never contains transcript or prompt text."""
        return None if self._last_metrics is None else dict(self._last_metrics)

    def token_cap(self, chunk: AudioChunk) -> int:
        """Dynamic completion limit: short chunks cannot legitimately need the full budget."""
        seconds = max(0.0, chunk.end - chunk.start)
        dynamic = math.ceil(self.min_tokens + self.tokens_per_second * seconds)
        return max(1, min(self.max_tokens, dynamic))

    def _resolve_temperature(self, temperature: float | None) -> float:
        if temperature is None:
            return self.temperature
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be a finite number between 0 and 2")
        return float(temperature)

    def forced_language(self, language: str | None) -> str | None:
        """Qwen3-ASR language name used as output prefix, or None for auto-detection."""
        if self.response_mode != "qwen3-asr" or not self.force_language:
            return None
        return qwen3_asr_language(language)

    def _interpret(
        self,
        response: dict,
        chunk: AudioChunk,
        prompt: str | None,
        outcome: dict,
        forced: str | None = None,
    ) -> str:
        """Return the validated transcript text ('' for explicit no speech)."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise DegenerateOutputError(
                f"llama.cpp returned no completion for audio chunk {chunk.index}."
            )
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            limit = self._last_metrics["max_tokens_requested"] if self._last_metrics else "?"
            raise DegenerateOutputError(
                f"Transcription of audio chunk {chunk.index} reached its token limit ({limit}); "
                "the transcript may be incomplete."
            )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if content is None:
            raise DegenerateOutputError(
                f"llama.cpp returned no content for audio chunk {chunk.index}. "
                "Confirm that the loaded model and projector support audio input."
            )
        if self.response_mode == "json":
            text = _extract_json_transcript(content, chunk.index)
        elif self.response_mode == "qwen3-asr":
            if forced:
                outcome["forced_language"] = forced
                text, detected = _parse_forced_qwen3_asr(content, chunk.index, forced)
            else:
                text, detected = _parse_qwen3_asr(content, chunk.index)
            outcome["detected_language"] = detected
        else:
            text = _extract_text(content, chunk.index)
        if not text:
            return ""
        ratio = _compression_ratio(text)
        if ratio is not None:
            outcome["compression_ratio"] = round(ratio, 3)
            if self._last_metrics is not None:
                self._last_metrics["compression_ratio"] = round(ratio, 3)
            if ratio > self.compression_ratio_threshold:
                raise DegenerateOutputError(
                    f"Transcript of audio chunk {chunk.index} looks like a repetition loop "
                    f"(compression ratio {ratio:.2f} > {self.compression_ratio_threshold:g})."
                )
        if prompt and _echoes_prompt(text, prompt):
            outcome["prompt_echo"] = True
            raise DegenerateOutputError(
                f"Transcript of audio chunk {chunk.index} repeats the instruction prompt "
                "instead of transcribing speech."
            )
        return text

    def effective_prompt(self, language: str | None) -> str | None:
        """Return the exact instruction submitted for output provenance.

        Qwen3-ASR mode sends the configured prompt only as system context and
        has no built-in prompt, so this is ``None`` without a configured prompt.
        """
        if self.prompt is not None:
            return self.prompt.strip()
        return self.builtin_prompt(language)

    def builtin_prompt(self, language: str | None) -> str | None:
        """Instruction used without a configured prompt (``None`` in Qwen3-ASR mode)."""
        if self.response_mode == "qwen3-asr":
            return None
        where = f" ({language})" if language else ""
        if self.response_mode == "json":
            return (
                f"Trascrivi letteralmente il parlato nell'audio nella lingua originale{where}. "
                "Nel campo transcript scrivi soltanto le parole pronunciate, in ordine. "
                "Nessuna introduzione, commento, traduzione o nome inventato."
            )
        return (
            f"Trascrivi letteralmente il parlato nell'audio nella lingua originale{where}. "
            "Restituisci soltanto le parole pronunciate, in ordine, senza introduzioni, "
            "commenti, traduzioni o nomi inventati."
        )

    def _save_response_audit(
        self,
        chunk: AudioChunk,
        prompt: str | None,
        response: dict,
        *,
        temperature: float,
        max_tokens: int,
        outcome: dict,
    ) -> None:
        response_dir = Path(chunk.path).parent / "responses"
        base = f"{chunk.index:05d}{chunk.part}_t{round(temperature * 100):03d}"
        audit = {
            "schema_version": 2,
            "backend": self.name,
            "model": self._server_model_id,
            "chunk": {
                "index": chunk.index,
                "part": chunk.part,
                "start": chunk.start,
                "end": chunk.end,
            },
            "request": {
                "prompt": prompt,
                "temperature": temperature,
                "seed": self.seed,
                "max_tokens": max_tokens,
                "configured_max_tokens": self.max_tokens,
                "min_tokens": self.min_tokens,
                "tokens_per_second": self.tokens_per_second,
                "repeat_penalty": self.repeat_penalty,
                "dry_multiplier": self.dry_multiplier,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "response_mode": self.response_mode,
                "response_format": (
                    _transcript_json_schema() if self.response_mode == "json" else None
                ),
            },
            "response": response,
            "outcome": outcome,
        }
        temporary_path = None
        try:
            response_dir.mkdir(parents=True, exist_ok=True)
            # Raw responses are evidence: never replace an earlier attempt's file.
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{base}.", suffix=".tmp", dir=response_dir
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(audit, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            # Choose the name and move the file under one lock so concurrent writers never
            # pick the same destination.
            with _AUDIT_LOCK:
                destination = response_dir / f"{base}.json"
                repeat = 1
                while destination.exists():
                    repeat += 1
                    destination = response_dir / f"{base}_r{repeat}.json"
                replace_with_retry(temporary_path, destination)
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
            error_type = TransientBackendError if exc.code >= 500 else TranscriptionError
            raise error_type(
                f"Could not {purpose}: llama.cpp returned HTTP {exc.code}"
                f"{': ' + detail if detail else ''}. Check that the server is running with an "
                "audio-capable model and projector."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise TransientBackendError(
                f"Could not {purpose} at {request.full_url}: {reason}. "
                "Start llama-server and verify its host and port."
            ) from exc
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DegenerateOutputError(
                f"llama.cpp returned invalid JSON while trying to {purpose}."
            ) from exc
        if not isinstance(parsed, dict):
            raise DegenerateOutputError(
                f"llama.cpp returned an invalid response while trying to {purpose}."
            )
        if isinstance(parsed.get("error"), dict):
            detail = _error_detail(json.dumps(parsed["error"], ensure_ascii=False).encode("utf-8"))
            code = parsed["error"].get("code")
            error_type = (
                TransientBackendError
                if isinstance(code, int) and not isinstance(code, bool) and code >= 500
                else TranscriptionError
            )
            raise error_type(f"Could not {purpose}: {detail or 'server reported an error'}.")
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


def _number(container: object, key: str) -> int | float | None:
    value = container.get(key) if isinstance(container, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _response_metrics(response: dict, max_tokens: int) -> dict:
    usage = response.get("usage")
    timings = response.get("timings")
    choices = response.get("choices")
    finish = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        finish = reason if isinstance(reason, str) else None
    return {
        "prompt_tokens": _number(usage, "prompt_tokens"),
        "completion_tokens": _number(usage, "completion_tokens"),
        "finish_reason": finish,
        "max_tokens_requested": max_tokens,
        "prompt_ms": _number(timings, "prompt_ms"),
        "predicted_ms": _number(timings, "predicted_ms"),
        "predicted_per_second": _number(timings, "predicted_per_second"),
        "cache_n": _number(timings, "cache_n"),
        "compression_ratio": None,
        **_logprob_metrics(choices),
    }


def _logprob_metrics(choices: object) -> dict:
    """Probability summary of the generated tokens; null when the server sent no logprobs."""
    values: list[float] = []
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        logprobs = choices[0].get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        if isinstance(content, list):
            for item in content:
                value = _number(item, "logprob")
                if value is not None:
                    values.append(math.exp(min(0.0, value)))
    if not values:
        return {"mean_token_prob": None, "min_token_prob": None, "logprob_tokens": None}
    return {
        "mean_token_prob": round(sum(values) / len(values), 6),
        "min_token_prob": round(min(values), 6),
        "logprob_tokens": len(values),
    }


def _compression_ratio(text: str) -> float | None:
    """Whisper-style ratio of raw to compressed bytes; None for very short texts."""
    if len(text) < _MIN_COMPRESSION_CHARS:
        return None
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw))


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _echoes_prompt(text: str, prompt: str) -> bool:
    """True when the transcript contains a run of prompt words verbatim."""
    prompt_words = _normalized_words(prompt)
    words = _normalized_words(text)
    size = _PROMPT_ECHO_WORDS
    if len(prompt_words) < size or len(words) < size:
        return False
    grams = {tuple(prompt_words[i : i + size]) for i in range(len(prompt_words) - size + 1)}
    return any(tuple(words[i : i + size]) in grams for i in range(len(words) - size + 1))


def _extract_text(content: object, chunk_index: int) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type", "text") == "text"
        ]
        return "".join(part for part in parts if isinstance(part, str)).strip()
    raise DegenerateOutputError(
        f"llama.cpp returned non-text content for audio chunk {chunk_index}."
    )


def _parse_qwen3_asr(content: object, chunk_index: int) -> tuple[str, str | None]:
    """Parse ``language <Name><asr_text><text>``; return (text, detected language)."""
    raw = _extract_text(content, chunk_index) if not isinstance(content, str) else content
    matched = _QWEN3_ASR.fullmatch(raw)
    if matched is not None:
        language = matched.group("language").strip()
        text = matched.group("text").strip()
        if language.casefold() == "none":
            if text:
                raise DegenerateOutputError(
                    f"Qwen3-ASR reported no language but returned text for audio chunk "
                    f"{chunk_index}."
                )
            return "", None
        if not text:
            return "", language or None
        return text, language or None
    if _QWEN3_ASR_NONE.fullmatch(raw) is not None:
        return "", None
    raise DegenerateOutputError(
        f"llama.cpp returned content that does not match the Qwen3-ASR format for audio "
        f"chunk {chunk_index}."
    )


def _qwen3_asr_prefix(language: str) -> str:
    return f"language {language}<asr_text>"


def _parse_forced_qwen3_asr(content: object, chunk_index: int, language: str) -> tuple[str, str]:
    """Parse a continuation of the forced ``language <Name><asr_text>`` prefix.

    llama.cpp returns only the generated continuation; a server that echoes the
    prefix is accepted too. Any other language prefix is degenerate output.
    """
    raw = _extract_text(content, chunk_index) if not isinstance(content, str) else content
    matched = _QWEN3_ASR.fullmatch(raw)
    if matched is not None:
        if matched.group("language").strip().casefold() != language.casefold():
            raise DegenerateOutputError(
                f"Qwen3-ASR ignored the forced language for audio chunk {chunk_index}."
            )
        return matched.group("text").strip(), language
    return raw.strip(), language


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
        raise DegenerateOutputError(
            f"llama.cpp returned non-text JSON content for audio chunk {chunk_index}."
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DegenerateOutputError(
            f"llama.cpp returned invalid JSON for audio chunk {chunk_index}; "
            "the constrained transcript response could not be parsed."
        ) from exc
    if not isinstance(parsed, dict) or set(parsed) != {"transcript"}:
        raise DegenerateOutputError(
            f"llama.cpp returned an invalid transcript JSON object for audio chunk {chunk_index}."
        )
    transcript = parsed["transcript"]
    if not isinstance(transcript, str):
        raise DegenerateOutputError(
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
