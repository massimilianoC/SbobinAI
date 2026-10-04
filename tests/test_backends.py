"""llama.cpp adapter hardening: errors, detectors, token cap, Qwen3-ASR, audit files."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from audio_transcript.adapters.llamacpp import LlamaCppBackend, qwen3_asr_language
from audio_transcript.adapters.mock import MockBackend
from audio_transcript.adapters.nexa import NexaBackend
from audio_transcript.domain.models import (
    AudioChunk,
    DegenerateOutputError,
    TranscriptionError,
    TransientBackendError,
)

URLOPEN = "audio_transcript.adapters.llamacpp.urllib.request.urlopen"


class FakeResponse:
    def __init__(self, payload):
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body


def completion(content, finish="stop", **extra):
    return {
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        **extra,
    }


def json_content(text):
    return json.dumps({"transcript": text})


class BackendCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.wav = Path(self._directory.name) / "speech.wav"
        self.wav.write_bytes(b"wav bytes")

    def chunk(self, start=0.0, end=2.0, index=0, part=""):
        return AudioChunk(self.wav, start, end, index, part)

    def run_backend(self, backend, outcomes, *, chunk=None, language=None, temperature=None):
        """Run one transcribe call; ``outcomes`` replace the chat completion reply."""
        replies = []
        if backend._server_model_id is None:
            replies = [
                FakeResponse({"status": "ok"}),
                FakeResponse({"data": [{"id": backend.model}]}),
            ]
        replies.extend(
            item if isinstance(item, BaseException) else FakeResponse(item) for item in outcomes
        )
        with patch(URLOPEN, side_effect=replies) as urlopen:
            try:
                result = backend.transcribe(
                    chunk or self.chunk(), language=language, temperature=temperature
                )
            finally:
                self.urlopen = urlopen
        return result

    def payload(self):
        return json.loads(self.urlopen.call_args.args[0].data)

    def audits(self):
        return sorted((self.wav.parent / "responses").glob("*.json"))


class ErrorClassificationTests(BackendCase):
    def http_error(self, code):
        return HTTPError(
            "http://127.0.0.1:8088/v1/chat/completions",
            code,
            "status",
            {},
            io.BytesIO(b'{"error":{"message":"server said no"}}'),
        )

    def test_connection_timeout_and_5xx_are_transient(self):
        for failure in (
            URLError("connection refused"),
            TimeoutError("timed out"),
            ConnectionResetError("reset"),
            self.http_error(500),
            self.http_error(503),
        ):
            with self.subTest(failure=repr(failure)):
                with self.assertRaises(TransientBackendError):
                    self.run_backend(LlamaCppBackend(), [failure])

    def test_http_4xx_is_a_plain_configuration_error(self):
        with self.assertRaises(TranscriptionError) as caught:
            self.run_backend(LlamaCppBackend(), [self.http_error(400)])
        self.assertNotIsInstance(caught.exception, (TransientBackendError, DegenerateOutputError))

    def test_unusable_replies_are_degenerate(self):
        bodies = {
            "length": completion("partial", finish="length"),
            "invalid json body": b"<html>gateway</html>",
            "not an object": b"[1, 2]",
            "no choices": {"choices": []},
            "missing content": {"choices": [{"finish_reason": "stop", "message": {}}]},
            "null content": completion(None),
            "bad transcript json": completion("not json"),
            "extra keys": completion('{"transcript":"a","x":1}'),
            "wrong type": completion('{"transcript":3}'),
        }
        for label, body in bodies.items():
            with self.subTest(label), self.assertRaises(DegenerateOutputError):
                self.run_backend(LlamaCppBackend(), [body])

    def test_server_error_in_a_200_body_follows_its_code(self):
        with self.assertRaises(TransientBackendError):
            self.run_backend(LlamaCppBackend(), [{"error": {"code": 503, "message": "busy"}}])
        with self.assertRaises(TranscriptionError) as caught:
            self.run_backend(LlamaCppBackend(), [{"error": {"code": 400, "message": "bad"}}])
        self.assertNotIsInstance(caught.exception, TransientBackendError)


class DetectorTests(BackendCase):
    def test_repetition_loop_is_rejected_but_the_audit_is_kept(self):
        looping = "grazie a tutti " * 30
        with self.assertRaisesRegex(DegenerateOutputError, "repetition loop"):
            self.run_backend(LlamaCppBackend(), [completion(json_content(looping))])
        (audit,) = self.audits()
        document = json.loads(audit.read_text(encoding="utf-8"))
        self.assertEqual(document["outcome"]["status"], "degenerate")
        self.assertGreater(document["outcome"]["compression_ratio"], 2.4)
        self.assertEqual(
            json.loads(document["response"]["choices"][0]["message"]["content"])["transcript"],
            looping,
        )

    def test_threshold_is_configurable(self):
        text = "uno due tre quattro cinque sei sette otto nove dieci " * 3
        backend = LlamaCppBackend(compression_ratio_threshold=100.0)
        segments = self.run_backend(backend, [completion(json_content(text))])
        self.assertEqual(segments[0].text, text.strip())

    def test_short_and_ordinary_text_is_not_flagged(self):
        for text in (
            "no no no no no no no no",
            "Sì sì sì",
            "Buongiorno a tutti e grazie di essere venuti a questa presentazione di oggi.",
        ):
            with self.subTest(text=text):
                segments = self.run_backend(LlamaCppBackend(), [completion(json_content(text))])
                self.assertEqual(segments[0].text, text)

    def test_prompt_echo_is_rejected_for_built_in_and_custom_prompts(self):
        backend = LlamaCppBackend()
        echo = backend.effective_prompt("it")
        with self.assertRaisesRegex(DegenerateOutputError, "instruction prompt"):
            self.run_backend(backend, [completion(json_content(echo))], language="it")
        partial = "Nessuna introduzione, commento, traduzione o nome inventato."
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(backend, [completion(json_content(partial))])

        custom = LlamaCppBackend(prompt="Write exactly the words that are spoken in the audio.")
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(
                custom,
                [completion(json_content("He said: write exactly the words that are spoken."))],
            )
        segments = self.run_backend(
            custom, [completion(json_content("Write exactly what you hear."))]
        )
        self.assertEqual(len(segments), 1)

    def test_five_shared_words_are_not_an_echo(self):
        backend = LlamaCppBackend(prompt="Transcribe the spoken words in the audio now")
        segments = self.run_backend(
            backend, [completion(json_content("please transcribe the spoken words in time"))]
        )
        self.assertEqual(len(segments), 1)

    def test_language_hint_is_folded_into_the_prompt_without_an_extra_sentence(self):
        backend = LlamaCppBackend()
        self.run_backend(backend, [completion(json_content("ciao"))], language="it")
        text = self.payload()["messages"][0]["content"][0]["text"]
        self.assertEqual(text, backend.effective_prompt("it"))
        self.assertNotIn("Lingua indicata", text)
        self.assertNotIn("Mantieni comunque", text)
        self.assertIn("(it)", text)
        self.assertNotIn("(", backend.effective_prompt(None))


class NoSpeechTests(BackendCase):
    def test_empty_json_and_plain_replies_mean_no_speech(self):
        self.assertEqual(
            self.run_backend(LlamaCppBackend(), [completion('{"transcript": ""}')]), []
        )
        (audit,) = self.audits()
        outcome = json.loads(audit.read_text(encoding="utf-8"))["outcome"]
        self.assertEqual(outcome["status"], "no_speech")
        plain = LlamaCppBackend(response_mode="plain")
        self.assertEqual(self.run_backend(plain, [completion("   ")]), [])
        self.assertEqual(self.run_backend(plain, [completion("")]), [])


class TokenCapTests(BackendCase):
    def test_dynamic_cap_scales_with_duration_and_never_exceeds_max_tokens(self):
        backend = LlamaCppBackend()
        self.assertEqual(backend.token_cap(self.chunk(0, 10)), 128)
        self.assertEqual(backend.token_cap(self.chunk(0, 2)), 64)
        self.assertEqual(backend.token_cap(self.chunk(0, 0.5)), 52)
        self.assertEqual(backend.token_cap(self.chunk(0, 1000)), 1024)
        tuned = LlamaCppBackend(max_tokens=100, min_tokens=10, tokens_per_second=2.5)
        self.assertEqual(tuned.token_cap(self.chunk(0, 3)), 18)
        self.assertEqual(tuned.token_cap(self.chunk(0, 60)), 100)

    def test_payload_uses_the_cap_and_the_audit_records_both_limits(self):
        self.run_backend(
            LlamaCppBackend(), [completion(json_content("ciao"))], chunk=self.chunk(0, 10)
        )
        self.assertEqual(self.payload()["max_tokens"], 128)
        (audit,) = self.audits()
        request = json.loads(audit.read_text(encoding="utf-8"))["request"]
        self.assertEqual(request["max_tokens"], 128)
        self.assertEqual(request["configured_max_tokens"], 1024)
        self.assertEqual(request["min_tokens"], 48)
        self.assertEqual(request["tokens_per_second"], 8.0)
        self.assertEqual(request["compression_ratio_threshold"], 2.4)

    def test_repetition_controls_are_sent_only_when_enabled(self):
        self.run_backend(LlamaCppBackend(), [completion(json_content("ciao"))])
        self.assertNotIn("repeat_penalty", self.payload())
        self.assertNotIn("dry_multiplier", self.payload())
        tuned = LlamaCppBackend(repeat_penalty=1.15, dry_multiplier=0.8)
        self.run_backend(tuned, [completion(json_content("ciao"))])
        self.assertEqual(self.payload()["repeat_penalty"], 1.15)
        self.assertEqual(self.payload()["dry_multiplier"], 0.8)
        (audit,) = (p for p in self.audits() if "_r" not in p.name)
        request = json.loads(audit.read_text(encoding="utf-8"))["request"]
        self.assertEqual((request["repeat_penalty"], request["dry_multiplier"]), (1.0, 0.0))

    def test_new_setting_validation(self):
        for kwargs in (
            {"min_tokens": 0},
            {"tokens_per_second": -1},
            {"compression_ratio_threshold": 0},
            {"repeat_penalty": 0},
            {"dry_multiplier": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LlamaCppBackend(**kwargs)


class AuditFileTests(BackendCase):
    def test_every_attempt_gets_its_own_audit_file(self):
        backend = LlamaCppBackend()
        part_chunk = self.chunk(1.0, 2.0, index=3, part="b")
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(backend, [completion("cut", finish="length")], chunk=part_chunk)
        self.run_backend(
            backend, [completion(json_content("ciao"))], chunk=part_chunk, temperature=0.4
        )
        self.run_backend(backend, [completion(json_content("ciao"))], chunk=part_chunk)
        self.run_backend(backend, [completion(json_content("ciao"))], chunk=part_chunk)
        names = [path.name for path in self.audits()]
        self.assertEqual(
            names,
            ["00003b_t000.json", "00003b_t000_r2.json", "00003b_t000_r3.json", "00003b_t040.json"],
        )
        first = json.loads((self.wav.parent / "responses" / "00003b_t000.json").read_text("utf-8"))
        self.assertEqual(first["response"]["choices"][0]["finish_reason"], "length")
        self.assertEqual(first["chunk"]["part"], "b")
        fallback = json.loads(
            (self.wav.parent / "responses" / "00003b_t040.json").read_text("utf-8")
        )
        self.assertEqual(fallback["request"]["temperature"], 0.4)

    def test_temperature_override_is_sent_and_validated(self):
        self.run_backend(LlamaCppBackend(), [completion(json_content("ciao"))], temperature=0.2)
        self.assertEqual(self.payload()["temperature"], 0.2)
        with self.assertRaises(ValueError):
            self.run_backend(LlamaCppBackend(), [], temperature=2.5)


class MetricsTests(BackendCase):
    def test_metrics_come_from_usage_and_timings_and_never_include_text(self):
        reply = completion(
            json_content("ciao a tutti da qui"),
            usage={"prompt_tokens": 90, "completion_tokens": 12},
            timings={
                "prompt_ms": 31.5,
                "predicted_ms": 480.0,
                "predicted_per_second": 25.0,
                "cache_n": 80,
            },
        )
        backend = LlamaCppBackend()
        self.assertIsNone(backend.last_call_metrics())
        self.run_backend(backend, [reply], chunk=self.chunk(0, 10))
        metrics = backend.last_call_metrics()
        self.assertEqual(metrics["prompt_tokens"], 90)
        self.assertEqual(metrics["completion_tokens"], 12)
        self.assertEqual(metrics["finish_reason"], "stop")
        self.assertEqual(metrics["max_tokens_requested"], 128)
        self.assertEqual(metrics["predicted_per_second"], 25.0)
        self.assertEqual(metrics["cache_n"], 80)
        self.assertIsNone(metrics["compression_ratio"])  # text too short to score
        self.assertNotIn("ciao", json.dumps(metrics))

    def test_metrics_are_set_on_degenerate_errors_and_reset_each_call(self):
        backend = LlamaCppBackend()
        loop = completion(
            json_content("grazie a tutti " * 30),
            usage={"completion_tokens": 120},
        )
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(backend, [loop])
        metrics = backend.last_call_metrics()
        self.assertEqual(metrics["completion_tokens"], 120)
        self.assertGreater(metrics["compression_ratio"], 2.4)
        with self.assertRaises(TransientBackendError):
            self.run_backend(backend, [URLError("down")])
        self.assertIsNone(backend.last_call_metrics())

    def test_missing_usage_and_timings_are_null(self):
        backend = LlamaCppBackend()
        self.run_backend(backend, [completion(json_content("ciao"))])
        metrics = backend.last_call_metrics()
        self.assertIsNone(metrics["completion_tokens"])
        self.assertIsNone(metrics["cache_n"])
        self.assertEqual(metrics["finish_reason"], "stop")


class Qwen3AsrTests(BackendCase):
    def backend(self, **kwargs):
        return LlamaCppBackend(model="qwen3-asr", response_mode="qwen3-asr", **kwargs)

    def test_request_contains_only_audio_and_no_response_format(self):
        backend = self.backend(force_language=False)
        self.run_backend(
            backend, [completion("language Italian<asr_text>Ciao a tutti")], language="it"
        )
        payload = self.payload()
        self.assertEqual([m["role"] for m in payload["messages"]], ["user"])
        content = payload["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["input_audio"])
        self.assertNotIn("response_format", payload)
        self.assertIsNone(backend.effective_prompt("it"))

    def test_configured_prompt_is_sent_as_system_context(self):
        backend = self.backend(prompt="Meeting about hydrology.")
        self.run_backend(backend, [completion("language English<asr_text>Hello there")])
        messages = self.payload()["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "Meeting about hydrology."})
        self.assertEqual([p["type"] for p in messages[1]["content"]], ["input_audio"])
        self.assertEqual(backend.effective_prompt(None), "Meeting about hydrology.")

    def test_valid_replies_are_parsed_strictly(self):
        cases = {
            "language Italian<asr_text>Ciao a tutti": "Ciao a tutti",
            "language English<asr_text>Line one\nline two": "Line one\nline two",
            "  language Chinese<asr_text>  你好  ": "你好",
        }
        for content, expected in cases.items():
            with self.subTest(content=content):
                segments = self.run_backend(self.backend(), [completion(content)])
                self.assertEqual([s.text for s in segments], [expected])
                self.assertEqual((segments[0].start, segments[0].end), (0.0, 2.0))

    def test_no_language_or_empty_text_means_no_speech(self):
        for content in (
            "language None<asr_text>",
            "language None",
            "language Italian<asr_text>",
            "language Italian<asr_text>   ",
        ):
            with self.subTest(content=content):
                self.assertEqual(self.run_backend(self.backend(), [completion(content)]), [])

    def test_anything_else_is_degenerate(self):
        for content in (
            "Ciao a tutti",
            "Italian<asr_text>Ciao",
            "language Italian Ciao a tutti",
            "language None<asr_text>invented words",
            "",
        ):
            with self.subTest(content=content):
                with self.assertRaises(DegenerateOutputError):
                    self.run_backend(self.backend(), [completion(content)])

    def test_detected_language_is_recorded_in_the_audit(self):
        self.run_backend(self.backend(), [completion("language Italian<asr_text>Ciao a tutti")])
        (audit,) = self.audits()
        outcome = json.loads(audit.read_text(encoding="utf-8"))["outcome"]
        self.assertEqual(outcome["detected_language"], "Italian")
        self.assertEqual(outcome["status"], "ok")

    def test_prompt_echo_check_depends_on_a_configured_prompt(self):
        sentence = "alpha beta gamma delta epsilon zeta eta theta"
        reply = completion(f"language English<asr_text>{sentence}")
        self.assertEqual(len(self.run_backend(self.backend(), [reply])), 1)
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(self.backend(prompt=sentence), [reply])

    def test_length_and_loops_are_still_degenerate(self):
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(
                self.backend(), [completion("language English<asr_text>abc", finish="length")]
            )
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(
                self.backend(),
                [completion("language English<asr_text>" + "thank you " * 40)],
            )


class WithoutPromptTests(BackendCase):
    def call(self, backend, content, *, language=None):
        replies = [
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [{"id": backend.model}]}),
            FakeResponse(completion(content)),
        ]
        with patch(URLOPEN, side_effect=replies) as urlopen:
            result = backend.transcribe(self.chunk(), language=language, use_prompt=False)
        self.urlopen = urlopen
        return result

    def test_qwen3_asr_drops_the_system_context(self):
        backend = LlamaCppBackend(
            model="qwen3-asr", response_mode="qwen3-asr", prompt="Names: Rossi, Bianchi"
        )
        segments = self.call(backend, "hm.", language="it")
        roles = [m["role"] for m in self.payload()["messages"]]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual([s.text for s in segments], ["hm."])

    def test_json_mode_falls_back_to_the_builtin_instruction(self):
        backend = LlamaCppBackend(prompt="Custom instruction with names")
        self.call(backend, '{"transcript": "ciao"}', language="it")
        text = self.payload()["messages"][0]["content"][0]["text"]
        self.assertEqual(text, backend.builtin_prompt("it"))
        self.assertNotIn("Custom instruction", text)


class Qwen3AsrForcedLanguageTests(BackendCase):
    def backend(self, **kwargs):
        return LlamaCppBackend(model="qwen3-asr", response_mode="qwen3-asr", **kwargs)

    def test_configured_language_is_prefilled_as_the_output_prefix(self):
        segments = self.run_backend(self.backend(), [completion("Ciao a tutti")], language="it")
        messages = self.payload()["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[-1]["content"], "language Italian<asr_text>")
        self.assertEqual([s.text for s in segments], ["Ciao a tutti"])
        (audit,) = self.audits()
        outcome = json.loads(audit.read_text(encoding="utf-8"))["outcome"]
        self.assertEqual(outcome["forced_language"], "Italian")

    def test_language_names_and_codes_map_to_qwen3_asr_names(self):
        self.assertEqual(qwen3_asr_language("it"), "Italian")
        self.assertEqual(qwen3_asr_language("ITALIAN"), "Italian")
        self.assertEqual(qwen3_asr_language("yue"), "Cantonese")
        self.assertEqual(qwen3_asr_language("italiano"), "Italian")
        self.assertEqual(qwen3_asr_language("Español"), "Spanish")
        self.assertEqual(qwen3_asr_language("pt-BR"), "Portuguese")
        self.assertIsNone(qwen3_asr_language("xx"))
        self.assertIsNone(qwen3_asr_language(None))

    def test_forced_reply_echoing_the_prefix_is_accepted(self):
        segments = self.run_backend(
            self.backend(), [completion("language Italian<asr_text>Ciao")], language="it"
        )
        self.assertEqual([s.text for s in segments], ["Ciao"])

    def test_empty_continuation_is_no_speech_and_other_prefix_is_degenerate(self):
        self.assertEqual(self.run_backend(self.backend(), [completion("  ")], language="it"), [])
        with self.assertRaises(DegenerateOutputError):
            self.run_backend(
                self.backend(), [completion("language Chinese<asr_text>加油")], language="it"
            )

    def test_no_prefill_without_language_unknown_language_or_when_disabled(self):
        for kwargs, language in (({}, None), ({}, "xx"), ({"force_language": False}, "it")):
            with self.subTest(kwargs=kwargs, language=language):
                self.run_backend(
                    self.backend(**kwargs),
                    [completion("language Italian<asr_text>Ciao")],
                    language=language,
                )
                self.assertEqual([m["role"] for m in self.payload()["messages"]], ["user"])

    def test_other_modes_never_prefill(self):
        backend = LlamaCppBackend(response_mode="plain")
        self.run_backend(backend, [completion("Ciao")], language="it")
        self.assertEqual([m["role"] for m in self.payload()["messages"]], ["user"])


class OtherBackendTests(unittest.TestCase):
    def test_mock_accepts_temperature(self):
        chunk = AudioChunk(Path("x.wav"), 0.0, 1.0, 0)
        segments = MockBackend().transcribe(chunk, language=None, temperature=0.4)
        self.assertEqual(len(segments), 1)

    def test_nexa_ignores_temperature_without_changing_behaviour(self):
        class Inference:
            def __init__(self, **kwargs):
                pass

            def inference(self, path, prompt):
                return " spoken words "

            def cleanup(self):
                return None

        module = type("Module", (), {"NexaAudioLMInference": Inference})
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "m.gguf"
            projector = Path(directory) / "p.gguf"
            model.write_bytes(b"m")
            projector.write_bytes(b"p")
            backend = NexaBackend(local_path=model, projector_local_path=projector)
            with patch.object(NexaBackend, "_import_runtime", return_value=module):
                chunk = AudioChunk(Path(directory) / "c.wav", 0.0, 1.0, 0)
                segments = backend.transcribe(chunk, language=None, temperature=0.4)
        self.assertEqual(segments[0].text, "spoken words")


if __name__ == "__main__":
    unittest.main()
