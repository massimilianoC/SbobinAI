"""Token-probability confidence proxy and duration formatting (synthetic data)."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.adapters.llamacpp import LlamaCppBackend
from audio_transcript.application.pipeline import (
    TranscriptionPipeline,
    _execution_summary,
)
from audio_transcript.application.reporting import format_duration, format_timing, render_run_report
from audio_transcript.config import load_config
from audio_transcript.domain.models import Segment

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
    from .test_backends import BackendCase, completion, json_content
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeBackend, FakeProcessor, make_config
    from test_backends import BackendCase, completion, json_content


def _logprobs(*probabilities):
    return {
        "content": [
            {"token": "t", "logprob": math.log(p), "bytes": [116], "top_logprobs": []}
            for p in probabilities
        ]
    }


def _with_logprobs(reply, *probabilities):
    reply["choices"][0]["logprobs"] = _logprobs(*probabilities)
    return reply


class LogprobMetricsTests(BackendCase):
    def test_logprobs_are_requested_by_default_and_summarised(self):
        backend = LlamaCppBackend()
        reply = _with_logprobs(completion(json_content("ciao a tutti")), 1.0, 0.5, 0.25)
        self.run_backend(backend, [reply])
        self.assertTrue(self.payload()["logprobs"])
        self.assertIn("response_format", self.payload())  # JSON schema path still works
        metrics = backend.last_call_metrics()
        self.assertAlmostEqual(metrics["mean_token_prob"], (1.0 + 0.5 + 0.25) / 3, places=5)
        self.assertAlmostEqual(metrics["min_token_prob"], 0.25, places=5)
        self.assertEqual(metrics["logprob_tokens"], 3)
        self.assertNotIn("ciao", json.dumps(metrics))

    def test_missing_logprobs_are_null(self):
        backend = LlamaCppBackend()
        self.run_backend(backend, [completion(json_content("ciao"))])
        metrics = backend.last_call_metrics()
        self.assertIsNone(metrics["mean_token_prob"])
        self.assertIsNone(metrics["min_token_prob"])
        self.assertIsNone(metrics["logprob_tokens"])

    def test_logprobs_can_be_disabled(self):
        backend = LlamaCppBackend(collect_logprobs=False)
        self.run_backend(backend, [completion(json_content("ciao"))])
        self.assertNotIn("logprobs", self.payload())
        with self.assertRaises(ValueError):
            LlamaCppBackend(collect_logprobs="yes")

    def test_qwen3_prefill_path_requests_logprobs(self):
        backend = LlamaCppBackend(model="qwen3-asr", response_mode="qwen3-asr")
        reply = _with_logprobs(completion(" ciao a tutti"), 0.9, 0.8)
        self.run_backend(backend, [reply], language="it")
        payload = self.payload()
        self.assertTrue(payload["logprobs"])
        self.assertEqual(payload["messages"][-1]["role"], "assistant")
        self.assertAlmostEqual(backend.last_call_metrics()["mean_token_prob"], 0.85, places=5)

    def test_config_flag_is_part_of_job_identity(self):
        self.assertTrue(load_config(overrides={"backend": "llamacpp"}).collect_logprobs)
        base = {"backend": "llamacpp", "model": "m", "vad": "none"}
        fingerprints = set()
        for flag in (True, False):
            config = make_config(Path("."), **base, collect_logprobs=flag)
            fingerprints.add(TranscriptionPipeline(config, FakeProcessor())._fingerprint())
        self.assertEqual(len(fingerprints), 2)


class ConfidenceAggregationTests(unittest.TestCase):
    @staticmethod
    def _entry(mean, tokens=10, outcome="ok"):
        return {
            "stage": "base",
            "outcome": outcome,
            "wall_seconds": 1.0,
            "audio_seconds": 1.0,
            "metrics": {"mean_token_prob": mean, "logprob_tokens": tokens, "completion_tokens": 5},
        }

    def test_aggregate_per_chunk_means(self):
        seg = [{"start": 0, "end": 1, "text": "x"}]
        checkpoint = {
            "chunks": {str(i): seg for i in range(4)},
            "failed": {},
            "attempts": {
                "0": [self._entry(0.9)],
                "1": [self._entry(0.8)],
                "2": [self._entry(0.4)],
                "3": [self._entry(0.2, outcome="DegenerateOutputError"), self._entry(0.6)],
            },
        }
        confidence = _execution_summary(checkpoint, 4, 4.0)["confidence"]
        self.assertEqual(confidence["method"], "mean_token_probability")
        self.assertIs(confidence["calibrated"], False)
        self.assertAlmostEqual(confidence["mean"], (0.9 + 0.8 + 0.4 + 0.6) / 4, places=4)
        self.assertEqual(confidence["low_chunks"], 1)
        self.assertAlmostEqual(confidence["p10_chunk"], 0.4, places=4)

    def test_no_metrics_means_null(self):
        checkpoint = {"chunks": {"0": []}, "failed": {}, "attempts": {"0": [self._entry(None)]}}
        self.assertIsNone(_execution_summary(checkpoint, 1, 1.0)["confidence"])

    def test_confidence_reaches_report_and_source_json(self):
        class Scored(FakeBackend):
            name = "llamacpp"

            def transcribe(self, chunk, *, language, temperature=None):
                self.current = 0.9 if chunk.index == 0 else 0.3
                return [Segment(chunk.start, chunk.end, "words")]

            def last_call_metrics(self):
                return {
                    "completion_tokens": 3,
                    "mean_token_prob": self.current,
                    "logprob_tokens": 3,
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input").mkdir()
            source = root / "input" / "a.wav"
            source.write_bytes(b"synthetic")
            config = make_config(root, backend="llamacpp", model="m", archive_inputs=False)
            result = TranscriptionPipeline(
                config,
                FakeProcessor(chunk_count=2),
                Scored(),
                FileExporter(),
                status=lambda _: None,
            ).run([source])[0]
            folder, version = result["job_id"].split("/")
            report = json.loads(
                (root / "output" / folder / version / "report.json").read_text("utf-8")
            )
            self.assertEqual(report["execution"]["confidence"]["low_chunks"], 1)
            info = json.loads((root / "output" / folder / "source.json").read_text("utf-8"))
            self.assertEqual(info["versions"][0]["confidence"]["low_chunks"], 1)
            self.assertIs(info["versions"][0]["confidence"]["calibrated"], False)
            run_report = (root / "output" / folder / version / "run-report.md").read_text("utf-8")
            self.assertIn("uncalibrated", run_report)


class DurationFormatTests(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(format_duration(0.4), "0.4 s")
        self.assertEqual(format_duration(0.048), "48 ms")
        self.assertEqual(format_duration(0.0), "0 ms")
        self.assertEqual(format_duration(59.9), "59.9 s")
        self.assertEqual(format_duration(61), "1 min 01 s")
        self.assertEqual(format_duration(3600), "1 h 00 min 00 s")
        self.assertEqual(format_duration(3903), "1 h 05 min 03 s")
        self.assertEqual(format_duration(262.4), "4 min 22 s")
        self.assertEqual(format_duration(59.96), "1 min 00 s")
        self.assertEqual(format_duration(None), "n/a")

    def test_format_timing_shows_human_and_exact_seconds(self):
        self.assertEqual(format_timing(262.4), "4 min 22 s (262.4 s)")
        self.assertEqual(format_timing(3903.0), "1 h 05 min 03 s (3903.0 s)")
        self.assertEqual(format_timing(0.4), "0.4 s")

    def test_run_report_has_required_sections_and_no_text(self):
        meta = {
            "source_name": "talk.wav",
            "version_folder": "20261002T1105Z_m_full_ab12cd34",
            "model": "m",
            "response_mode": "json",
            "language": "it",
            "scope": {"kind": "bounded", "max_duration_seconds": 600.0},
            "status": "completed",
            "audio_metadata": {"duration_seconds": 600.0, "speech_seconds": 300.0},
            "runs": [
                {
                    "started_at": "2026-10-02T11:05:00+00:00",
                    "ended_at": "2026-10-02T11:09:22+00:00",
                    "wall_seconds": 262.4,
                    "status": "completed",
                    "stages_seconds": {"preparation": 49.4, "inference": 188.0, "export": 6.1},
                }
            ],
        }
        execution = {
            "total_chunks": 3,
            "chunks_ok": 3,
            "chunks_no_speech": 0,
            "chunks_failed": 0,
            "analysed_audio_seconds": 600.0,
            "real_time_factor": 0.5,
            "completion_tokens": {"sum": 30, "max": 12, "p95": 12},
            "chunk_inference_seconds": {"mean": 62.0, "p95": 70.0, "max": 70.0},
            "temperature_fallbacks_used": 0,
            "splits_used": 0,
            "confidence": None,
            "stages_seconds": {"preparation": 49.4, "inference": 188.0, "export": 6.1},
        }
        text = render_run_report(meta, execution, completion_percent=100.0)
        for expected in (
            "talk.wav",
            "first 600 s (bounded run)",
            "4 min 22 s (262.4 s)",
            "3 min 08 s (188.0 s)",
            "2.0x faster than real time",
            "2026-10-02T11:05:00+00:00",
            "Confidence: unavailable",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
