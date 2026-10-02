import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from audio_transcript.adapters.llamacpp import LlamaCppBackend
from audio_transcript.domain.models import AudioChunk, TranscriptionError


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body


class LlamaCppBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = LlamaCppBackend()

    def _health_and_models(self, model=None):
        model = model or {"id": "qwen2-audio-7b"}
        return [
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [model]}),
        ]

    def test_check_health_and_loaded_model(self):
        with patch(
            "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
            side_effect=self._health_and_models(),
        ) as urlopen:
            self.backend.check()
        self.assertEqual(self.backend._server_model_id, "qwen2-audio-7b")
        self.assertEqual(
            [call.args[0].full_url for call in urlopen.call_args_list],
            ["http://127.0.0.1:8088/health", "http://127.0.0.1:8088/v1/models"],
        )

    def test_check_rejects_model_metadata_without_audio(self):
        metadata = {
            "id": "qwen2-audio-7b",
            "meta": {"architecture": {"input_modalities": ["text", "image"]}},
        }
        with patch(
            "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
            side_effect=self._health_and_models(metadata),
        ):
            with self.assertRaisesRegex(TranscriptionError, "does not advertise audio"):
                self.backend.check()

    def test_check_does_not_substitute_a_different_single_model(self):
        with patch(
            "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
            side_effect=self._health_and_models({"id": "another-model"}),
        ):
            with self.assertRaisesRegex(TranscriptionError, "not loaded"):
                self.backend.check()

    def test_transcribe_posts_base64_wav_and_returns_coarse_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "speech.wav"
            wav_bytes = b"RIFF\x00\x00test wav data"
            wav.write_bytes(wav_bytes)
            chunk = AudioChunk(wav, start=3.25, end=5.0, index=4)
            responses = self._health_and_models() + [
                FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": '{"transcript":"  Ciao, questa è una prova.  "}'
                                },
                            }
                        ]
                    }
                )
            ]
            with patch(
                "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
                side_effect=responses,
            ) as urlopen:
                segments = self.backend.transcribe(chunk, language="Italian")
            audit_path = wav.parent / "responses" / "00004.json"
            audit_text = audit_path.read_text(encoding="utf-8")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        content = payload["messages"][0]["content"]
        self.assertEqual(request.full_url, "http://127.0.0.1:8088/v1/chat/completions")
        self.assertEqual(payload["model"], "qwen2-audio-7b")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(
            payload["response_format"],
            {
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
            },
        )
        self.assertEqual(content[1]["type"], "input_audio")
        self.assertEqual(content[1]["input_audio"]["format"], "wav")
        self.assertEqual(base64.b64decode(content[1]["input_audio"]["data"]), wav_bytes)
        self.assertIn("lingua originale", content[0]["text"])
        self.assertIn("campo transcript", content[0]["text"])
        self.assertIn("Italian", content[0]["text"])
        self.assertIn("nome inventato", content[0]["text"])
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start, segments[0].end), (3.25, 5.0))
        self.assertEqual(segments[0].text, "Ciao, questa è una prova.")
        self.assertIsNone(segments[0].speaker)
        self.assertEqual(segments[0].timing_source, "chunk")
        audit = json.loads(audit_text)
        self.assertEqual(
            audit["response"]["choices"][0]["message"]["content"],
            '{"transcript":"  Ciao, questa è una prova.  "}',
        )
        self.assertEqual(audit["request"]["seed"], 42)
        self.assertNotIn(content[1]["input_audio"]["data"], audit_text)

    def test_length_finish_reason_fails_instead_of_returning_truncated_text(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "speech.wav"
            wav.write_bytes(b"wav")
            responses = self._health_and_models() + [
                FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "incomplete words"},
                            }
                        ]
                    }
                )
            ]
            with patch(
                "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
                side_effect=responses,
            ):
                with self.assertRaisesRegex(TranscriptionError, "may be incomplete"):
                    self.backend.transcribe(AudioChunk(wav, start=0, end=2, index=0), language=None)
            audit = json.loads(
                (wav.parent / "responses" / "00000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["response"]["choices"][0]["finish_reason"], "length")
            self.assertEqual(audit["request"]["response_format"]["type"], "json_schema")

    def test_http_audio_rejection_has_actionable_error(self):
        error = HTTPError(
            "http://127.0.0.1:8088/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"The current model does not support audio input"}}'),
        )
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "speech.wav"
            wav.write_bytes(b"wav")
            with patch(
                "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
                side_effect=self._health_and_models() + [error],
            ):
                with self.assertRaisesRegex(
                    TranscriptionError, "does not support audio input.*audio-capable model"
                ):
                    self.backend.transcribe(AudioChunk(wav, start=0, end=1, index=0), language=None)

    def test_empty_response_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "speech.wav"
            wav.write_bytes(b"wav")
            responses = self._health_and_models() + [
                FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": '{"transcript":"  "}'},
                            }
                        ]
                    }
                )
            ]
            with patch(
                "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
                side_effect=responses,
            ):
                with self.assertRaisesRegex(TranscriptionError, "empty transcript"):
                    self.backend.transcribe(AudioChunk(wav, 0, 1, 0), language=None)

    def test_json_mode_rejects_malformed_or_wrong_shape_content(self):
        invalid_content = [
            "not json",
            '{"transcript":"ok","extra":"not allowed"}',
            '{"transcript":12}',
        ]
        for content in invalid_content:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                wav = Path(directory) / "speech.wav"
                wav.write_bytes(b"wav")
                responses = self._health_and_models() + [
                    FakeResponse(
                        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
                    )
                ]
                with patch(
                    "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
                    side_effect=responses,
                ):
                    with self.assertRaises(TranscriptionError):
                        self.backend.transcribe(AudioChunk(wav, 0, 1, 0), language=None)

    def test_plain_mode_custom_prompt_and_deterministic_parameters(self):
        backend = LlamaCppBackend(
            prompt="Transcribe words only.",
            temperature=0.35,
            seed=2026,
            response_mode="plain",
        )
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "speech.wav"
            wav_bytes = b"wav bytes"
            wav.write_bytes(wav_bytes)
            responses = self._health_and_models() + [
                FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": " parole originali "},
                            }
                        ]
                    }
                )
            ]
            with patch(
                "audio_transcript.adapters.llamacpp.urllib.request.urlopen",
                side_effect=responses,
            ) as urlopen:
                segments = backend.transcribe(AudioChunk(wav, 0, 1, 0), language="Italian")
            payload = json.loads(urlopen.call_args.args[0].data)
            self.assertNotIn("response_format", payload)
            self.assertEqual(payload["temperature"], 0.35)
            self.assertEqual(payload["seed"], 2026)
            prompt = payload["messages"][0]["content"][0]["text"]
            self.assertTrue(prompt.startswith("Transcribe words only."))
            self.assertIn("Lingua indicata: Italian", prompt)
            self.assertEqual(segments[0].text, "parole originali")
            audit_path = wav.parent / "responses" / "00000.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["request"]["response_mode"], "plain")
            self.assertIsNone(audit["request"]["response_format"])
            self.assertNotIn(
                base64.b64encode(wav_bytes).decode("ascii"), audit_path.read_text(encoding="utf-8")
            )

    def test_prompt_and_sampling_settings_are_validated(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            LlamaCppBackend(temperature=2.1)
        with self.assertRaisesRegex(ValueError, "seed"):
            LlamaCppBackend(seed=-1)
        with self.assertRaisesRegex(ValueError, "response_mode"):
            LlamaCppBackend(response_mode="xml")

    def test_base_url_requires_loopback_http(self):
        with self.assertRaisesRegex(ValueError, "local loopback"):
            LlamaCppBackend(base_url="https://api.example.com")
        backend = LlamaCppBackend(base_url="http://localhost:8088/v1/")
        self.assertEqual(backend.base_url, "http://localhost:8088")


if __name__ == "__main__":
    unittest.main()
