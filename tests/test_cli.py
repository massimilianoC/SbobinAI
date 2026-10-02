"""CLI error contracts without model downloads or private fixtures."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript.cli import _doctor, _make_detector, _make_processor, main
from audio_transcript.config import load_config


class CliTests(unittest.TestCase):
    def test_invalid_config_returns_usage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.toml"
            config.write_text('sample_rate = "invalid"\n', encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["run", "--config", str(config)])
            self.assertEqual(code, 2)
            self.assertIn("sample_rate", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_missing_runtime_fails_before_media_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "generic.wav").touch()
            stderr = io.StringIO()
            with patch("audio_transcript.cli._make_processor") as factory:
                with contextlib.redirect_stderr(stderr):
                    code = main(
                        [
                            "run",
                            "--input-dir",
                            str(inputs),
                            "--process-dir",
                            str(root / "process"),
                            "--output-dir",
                            str(root / "output"),
                        ]
                    )
            self.assertEqual(code, 1)
            factory.return_value.prepare.assert_not_called()
            self.assertIn("Nexa", stderr.getvalue())

    def _run_with_results(self, results, *extra):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = {}

            class FakePipeline:
                def __init__(self, config, *args, **kwargs):
                    captured["config"] = config

                def run(self, sources, limit=None):
                    return results

            with (
                patch("audio_transcript.cli.TranscriptionPipeline", FakePipeline),
                patch("audio_transcript.cli._make_processor"),
                patch("audio_transcript.cli._make_backend"),
                patch("audio_transcript.cli._make_exporter"),
            ):
                code = main(
                    [
                        "run",
                        "--backend",
                        "mock",
                        "--input-dir",
                        str(root / "input"),
                        "--process-dir",
                        str(root / "process"),
                        "--output-dir",
                        str(root / "output"),
                        "--processed-dir",
                        str(root / "processed"),
                        *extra,
                    ]
                )
        return code, captured["config"]

    def test_failed_and_incomplete_jobs_exit_nonzero(self):
        for statuses, expected in (
            (["completed", "skipped", "prepared"], 0),
            (["completed", "failed"], 1),
            (["completed", "incomplete"], 1),
        ):
            with self.subTest(statuses=statuses):
                code, _ = self._run_with_results([{"status": s} for s in statuses])
                self.assertEqual(code, expected)

    def test_new_flags_override_configuration(self):
        _, config = self._run_with_results(
            [],
            "--vad",
            "energy",
            "--vad-model-path",
            "custom.onnx",
            "--vad-threshold",
            "0.6",
            "--vad-min-speech-seconds",
            "0.3",
            "--vad-min-silence-seconds",
            "0.2",
            "--vad-speech-pad-seconds",
            "0.1",
            "--vad-max-merge-gap-seconds",
            "2",
            "--vad-energy-margin-db",
            "12",
            "--fallback-temperatures",
            "0.1, 0.5",
            "--no-split-on-failure",
            "--min-tokens",
            "32",
            "--tokens-per-second",
            "6",
            "--compression-ratio-threshold",
            "2.0",
            "--repeat-penalty",
            "1.1",
            "--dry-multiplier",
            "0.8",
            "--response-mode",
            "qwen3-asr",
            "--chunk-seconds",
            "20",
        )
        self.assertEqual(config.vad, "energy")
        self.assertEqual(config.vad_model_path, Path("custom.onnx"))
        self.assertEqual(config.vad_threshold, 0.6)
        self.assertEqual(config.vad_min_speech_seconds, 0.3)
        self.assertEqual(config.vad_max_merge_gap_seconds, 2.0)
        self.assertEqual(config.vad_energy_margin_db, 12.0)
        self.assertEqual(config.fallback_temperatures, (0.1, 0.5))
        self.assertFalse(config.split_on_failure)
        self.assertEqual((config.min_tokens, config.tokens_per_second), (32, 6.0))
        self.assertEqual(config.compression_ratio_threshold, 2.0)
        self.assertEqual((config.repeat_penalty, config.dry_multiplier), (1.1, 0.8))
        self.assertEqual(config.response_mode, "qwen3-asr")
        self.assertEqual(config.chunk_seconds, 20.0)

    def test_parallel_and_intermediate_flags_override_configuration(self):
        _, config = self._run_with_results(
            [],
            "--backend",
            "llamacpp",
            "--parallel-requests",
            "3",
            "--intermediate-interval-seconds",
            "2.5",
        )
        self.assertEqual(config.parallel_requests, 3)
        self.assertEqual(config.intermediate_interval_seconds, 2.5)

    def test_empty_fallback_list_disables_temperature_fallback(self):
        _, config = self._run_with_results([], "--fallback-temperatures", "")
        self.assertEqual(config.fallback_temperatures, ())

    def test_invalid_fallback_values_return_usage_errors(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["run", "--fallback-temperatures", "hot"])
        self.assertEqual(caught.exception.code, 2)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["run", "--fallback-temperatures", "3.5"])
        self.assertEqual(code, 2)
        self.assertIn("fallback", stderr.getvalue())

    def test_processor_receives_the_configured_detector(self):
        config = load_config(
            overrides={"vad": "energy", "vad_threshold": 0.7, "vad_energy_margin_db": 9.0}
        )
        calls = {}

        def make_detector(name, **kwargs):
            calls["detector"] = (name, kwargs)
            return "detector-object"

        class Processor:
            def __init__(self, **kwargs):
                calls["processor"] = kwargs

        module = type(
            "Module",
            (),
            {"make_detector": staticmethod(make_detector), "FFmpegProcessor": Processor},
        )
        with patch("audio_transcript.cli.importlib.import_module", lambda *a, **k: module):
            _make_processor(config)
        name, kwargs = calls["detector"]
        self.assertEqual(name, "energy")
        self.assertEqual(kwargs["threshold"], 0.7)
        self.assertEqual(kwargs["energy_margin_db"], 9.0)
        self.assertEqual(kwargs["model_path"], config.vad_model_path)
        self.assertEqual(kwargs["ffmpeg"], "ffmpeg")
        self.assertEqual(calls["processor"]["detector"], "detector-object")
        self.assertEqual(calls["processor"]["ffprobe"], "ffprobe")

    def test_no_detector_is_passed_through_as_none(self):
        config = load_config(overrides={"vad": "none"})
        module = type("Module", (), {"make_detector": staticmethod(lambda name, **kw: None)})
        with patch("audio_transcript.cli.importlib.import_module", lambda *a, **k: module):
            self.assertIsNone(_make_detector(config))

    def test_doctor_reports_an_unusable_detector(self):
        class Detector:
            def check(self):
                raise RuntimeError("model file missing")

        class Backend:
            def check(self):
                return None

            def close(self):
                return None

        config = load_config(overrides={"backend": "mock"})
        stderr = io.StringIO()
        with (
            patch("audio_transcript.cli._make_detector", return_value=Detector()),
            patch("audio_transcript.cli._make_backend", return_value=Backend()),
            patch("audio_transcript.cli.shutil.which", return_value="ffmpeg"),
            patch(
                "audio_transcript.cli.subprocess.run",
                return_value=type("Result", (), {"returncode": 0, "stderr": ""})(),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(_doctor(config), 1)
        self.assertIn("Speech detector unavailable", stderr.getvalue())
        self.assertIn("model file missing", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
