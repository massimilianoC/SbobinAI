"""Core pipeline checks use synthetic chunks and no media or model dependencies."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application import pipeline as pipeline_module
from audio_transcript.application.pipeline import ProcessLock, TranscriptionPipeline, run_watch
from audio_transcript.cli import _doctor, _make_backend
from audio_transcript.config import load_config
from audio_transcript.domain.models import MediaInfo, Segment

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeBackend, FakeProcessor, make_config

_config = make_config


def _quiet(config, processor, backend, exporter=None):
    return TranscriptionPipeline(
        config, processor, backend, exporter or FileExporter(), status=lambda _: None
    )


class ConfigTests(unittest.TestCase):
    def test_config_overrides_toml_and_validates_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / "settings.toml"
            config_file.write_text('backend = "mock"\nchunk_seconds = 12\n', encoding="utf-8")
            config = load_config(config_file, {"chunk_seconds": 8.0})
            self.assertEqual(config.backend, "mock")
            self.assertEqual(config.model, "mock")
            self.assertEqual(config.chunk_seconds, 8)

            config_file.write_text('sample_rate = "16000"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample_rate"):
                load_config(config_file)

    def test_llamacpp_config_defaults_and_validation(self):
        config = load_config(
            overrides={
                "backend": "llamacpp",
                "base_url": "https://localhost:8088/v1",
                "timeout": 45.0,
                "max_tokens": 512,
                "temperature": 0.4,
                "seed": 123,
                "response_mode": "plain",
                "prompt": "Transcribe the speech.",
            }
        )
        self.assertEqual(config.model, "qwen2-audio-7b")
        self.assertEqual(config.base_url, "https://localhost:8088/v1")
        self.assertEqual(config.timeout, 45)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.temperature, 0.4)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.response_mode, "plain")
        self.assertEqual(config.prompt, "Transcribe the speech.")
        for invalid in (
            {"backend": "llamacpp", "base_url": "file:///tmp/model"},
            {"backend": "llamacpp", "timeout": 0},
            {"backend": "llamacpp", "max_tokens": 0},
            {"backend": "llamacpp", "max_tokens": True},
            {"backend": "llamacpp", "temperature": 2.1},
            {"backend": "llamacpp", "seed": -1},
            {"backend": "llamacpp", "response_mode": "xml"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_config(overrides=invalid)

    def test_new_defaults_and_qwen3_mode(self):
        config = load_config(overrides={"response_mode": "qwen3-asr"})
        self.assertEqual(config.response_mode, "qwen3-asr")
        self.assertEqual(config.chunk_seconds, 15.0)
        self.assertEqual(config.vad, "silero")
        self.assertEqual(config.vad_model_path, Path("models/silero_vad.onnx"))
        self.assertEqual(config.fallback_temperatures, (0.2, 0.4))
        self.assertTrue(config.split_on_failure)
        self.assertEqual((config.min_tokens, config.tokens_per_second), (48, 8.0))
        self.assertEqual(config.compression_ratio_threshold, 2.4)
        self.assertEqual((config.repeat_penalty, config.dry_multiplier), (1.0, 0.0))

    def test_new_settings_validation_and_toml_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.toml"
            path.write_text("fallback_temperatures = [0.1, 1]\n", encoding="utf-8")
            self.assertEqual(load_config(path).fallback_temperatures, (0.1, 1.0))
            path.write_text("fallback_temperatures = []\n", encoding="utf-8")
            self.assertEqual(load_config(path).fallback_temperatures, ())
        for invalid in (
            {"fallback_temperatures": [2.5]},
            {"fallback_temperatures": [-0.1]},
            {"fallback_temperatures": "0.2"},
            {"fallback_temperatures": [True]},
            {"vad": "webrtc"},
            {"vad_threshold": 1.0},
            {"vad_min_speech_seconds": -1.0},
            {"min_tokens": 0},
            {"tokens_per_second": -1.0},
            {"compression_ratio_threshold": 0.0},
            {"repeat_penalty": 0.0},
            {"dry_multiplier": -0.5},
            {"split_on_failure": "yes"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_config(overrides=invalid)

    def test_llamacpp_factory_and_doctor_use_selected_backend(self):
        class Backend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.checked = False

            def check(self):
                self.checked = True

            def close(self):
                return None

        with patch("audio_transcript.cli.importlib.import_module") as importer:
            importer.return_value = type("Module", (), {"LlamaCppBackend": Backend})
            config = load_config(
                overrides={
                    "backend": "llamacpp",
                    "base_url": "http://127.0.0.1:8088",
                    "timeout": 12.0,
                    "max_tokens": 256,
                }
            )
            backend = _make_backend(config)
        self.assertEqual(
            backend.kwargs,
            {
                "model": "qwen2-audio-7b",
                "base_url": "http://127.0.0.1:8088",
                "timeout": 12,
                "max_tokens": 256,
                "temperature": 0.0,
                "seed": 42,
                "response_mode": "json",
                "prompt": None,
                "min_tokens": 48,
                "tokens_per_second": 8.0,
                "compression_ratio_threshold": 2.4,
                "repeat_penalty": 1.0,
                "dry_multiplier": 0.0,
                "force_language": True,
                "collect_logprobs": True,
            },
        )

        backend = Backend()
        detector = type("Detector", (), {"checked": False})()
        detector.check = lambda: setattr(detector, "checked", True)
        with (
            patch("audio_transcript.cli._make_backend", return_value=backend),
            patch("audio_transcript.cli._make_detector", return_value=detector),
            patch("audio_transcript.cli.shutil.which", return_value="ffmpeg"),
            patch(
                "audio_transcript.cli.subprocess.run",
                return_value=type("Result", (), {"returncode": 0, "stderr": ""})(),
            ),
        ):
            self.assertEqual(_doctor(config), 0)
        self.assertTrue(backend.checked)
        self.assertTrue(detector.checked)


class RunTests(unittest.TestCase):
    def test_run_skips_completed_and_isolates_configuration_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "café audio.wav"
            source.parent.mkdir()
            source.write_bytes(b"original")
            processor = FakeProcessor(chunk_count=1)
            first = _quiet(_config(root), processor, FakeBackend())
            self.assertEqual(first.run()[0]["status"], "completed")
            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual(first.run()[0]["status"], "skipped")
            self.assertEqual(processor.prepare_calls, 1)

            changed = _quiet(_config(root, model="another-model"), processor, FakeBackend())
            self.assertEqual(changed.run()[0]["status"], "completed")
            folders = [p for p in (root / "output").iterdir() if p.is_dir()]
            self.assertEqual(len(folders), 1)
            versions = [p for p in folders[0].iterdir() if p.is_dir()]
            self.assertEqual(len(versions), 2)
            self.assertTrue(any("another-model" in p.name for p in versions))

    def test_real_backend_archives_only_direct_queued_input_after_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "spoken sample.wav"
            source.parent.mkdir()
            original = b"source media bytes"
            source.write_bytes(original)
            external = root / "external.wav"
            external.write_bytes(b"external media")
            backend = FakeBackend()
            backend.name = "llamacpp"
            config = _config(root, backend="llamacpp", model="qwen2-audio-7b")
            pipeline = _quiet(config, FakeProcessor(chunk_count=1), backend)
            result = pipeline.run([source, external])
            self.assertEqual([item["status"] for item in result], ["completed", "completed"])
            self.assertFalse(source.exists())
            archived = root / "processed" / result[0]["job_id"].split("/")[0] / source.name
            self.assertEqual(archived.read_bytes(), original)
            self.assertEqual(external.read_bytes(), b"external media")
            metadata = json.loads(
                (root / "process" / result[0]["job_id"] / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(metadata["archived_source_path"].endswith(source.name))
            self.assertFalse(any(path.is_dir() for path in (root / "input").iterdir()))

    def test_archive_disabled_mock_and_prepare_only_preserve_sources(self):
        for mode in ("mock", "disabled", "prepare-only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "input" / "keep.wav"
                source.parent.mkdir()
                source.write_bytes(b"retain original")
                if mode == "mock":
                    config = _config(root, backend="mock", model="mock")
                    backend = FakeBackend()
                elif mode == "disabled":
                    config = _config(
                        root, backend="llamacpp", model="qwen2-audio-7b", archive_inputs=False
                    )
                    backend = FakeBackend()
                    backend.name = "llamacpp"
                else:
                    config = _config(
                        root, backend="llamacpp", model="qwen2-audio-7b", prepare_only=True
                    )
                    backend = None
                pipeline = TranscriptionPipeline(
                    config,
                    FakeProcessor(chunk_count=1),
                    backend,
                    None if config.prepare_only else FileExporter(),
                    status=lambda _: None,
                )
                pipeline.run([source])
                if mode == "mock":
                    self.assertEqual(pipeline.run([source])[0]["status"], "skipped")
                self.assertEqual(source.read_bytes(), b"retain original")

    def test_archive_collision_keeps_completed_transcript_and_retries_later(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "collision.wav"
            source.parent.mkdir()
            source.write_bytes(b"original input")
            backend = FakeBackend()
            backend.name = "llamacpp"
            config = _config(root, backend="llamacpp", model="qwen2-audio-7b")
            pipeline = _quiet(config, FakeProcessor(chunk_count=1), backend)
            job_id, _ = pipeline._identity(source, pipeline._fingerprint())
            target = root / "processed" / job_id.split("/")[0] / source.name
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different data")
            result = pipeline.run([source])[0]
            self.assertEqual(result["status"], "completed")
            self.assertIn("archive_warning", result)
            self.assertEqual(source.read_bytes(), b"original input")
            self.assertEqual(target.read_bytes(), b"different data")
            metadata_path = root / "process" / job_id / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertIn("archive_warning", metadata)
            self.assertFalse((root / "process" / job_id / "failure.json").exists())
            resumed = pipeline.run([source])[0]
            self.assertEqual(resumed["status"], "skipped")
            self.assertIn("archive_warning", resumed)

            target.write_bytes(b"original input")
            resumed = pipeline.run([source])[0]
            self.assertEqual(resumed["status"], "skipped")
            self.assertFalse(source.exists())
            self.assertEqual(target.read_bytes(), b"original input")

    def test_bounded_real_backend_run_retains_full_source_and_does_not_archive_on_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "long-recording.wav"
            source.parent.mkdir()
            source.write_bytes(b"full source media")

            class BoundedProcessor(FakeProcessor):
                def probe(self, source):
                    return MediaInfo(duration=60.0, sample_rate=16000, channels=1, codec="fake")

            backend = FakeBackend()
            backend.name = "llamacpp"
            config = _config(root, backend="llamacpp", model="qwen2-audio-7b", max_duration=1.0)
            pipeline = _quiet(config, BoundedProcessor(chunk_count=1), backend)
            result = pipeline.run([source])[0]
            self.assertEqual(result["status"], "completed")
            self.assertIn("Only part of the source was processed", result["archive_warning"])
            self.assertEqual(source.read_bytes(), b"full source media")
            self.assertFalse(
                (root / "processed" / result["job_id"].split("/")[0] / source.name).exists()
            )

            resumed = pipeline.run([source])[0]
            self.assertEqual(resumed["status"], "skipped")
            self.assertIn("Only part of the source was processed", resumed["archive_warning"])
            self.assertEqual(source.read_bytes(), b"full source media")

    def test_llamacpp_generation_settings_are_recorded_and_isolate_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "settings.wav"
            source.parent.mkdir()
            source.write_bytes(b"original")
            backend = FakeBackend()
            backend.name = "llamacpp"
            options = {
                "backend": "llamacpp",
                "model": "qwen2-audio-7b",
                "archive_inputs": False,
                "temperature": 0.25,
                "seed": 17,
                "response_mode": "plain",
                "prompt": "Transcribe faithfully.",
            }
            first = _quiet(_config(root, **options), FakeProcessor(chunk_count=1), backend)
            result = first.run()[0]
            report = json.loads(
                (root / "output" / result["job_id"] / "report.json").read_text(encoding="utf-8")
            )
            inference = report["inference_configuration"]
            self.assertEqual(inference["temperature"], 0.25)
            self.assertEqual(inference["seed"], 17)
            self.assertEqual(inference["response_mode"], "plain")
            self.assertEqual(inference["prompt"], "Transcribe faithfully.")
            self.assertEqual(inference["fallback_temperatures"], [0.2, 0.4])
            self.assertEqual(inference["min_tokens"], 48)
            self.assertEqual(inference["repeat_penalty"], 1.0)

            second = _quiet(
                _config(root, **{**options, "temperature": 0.5}),
                FakeProcessor(chunk_count=1),
                FakeBackend(),
            )
            self.assertNotEqual(second.run([source])[0]["job_id"], result["job_id"])

    def test_effective_prompt_is_written_to_transcript_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "prompt.wav"
            source.parent.mkdir()
            source.write_bytes(b"audio fixture")

            class PromptBackend(FakeBackend):
                name = "llamacpp"

                def effective_prompt(self, language):
                    return f"Resolved Italian prompt ({language})"

            config = _config(
                root,
                backend="llamacpp",
                model="qwen2-audio-7b",
                language="it",
                archive_inputs=False,
            )
            pipeline = _quiet(config, FakeProcessor(chunk_count=1), PromptBackend())
            result = pipeline.run([source])[0]
            job_dir = root / "output" / result["job_id"]
            report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
            transcript = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
            for document in (report, transcript):
                inference = document["inference_configuration"]
                self.assertEqual(inference["prompt"], "Resolved Italian prompt (it)")
                self.assertIsNone(inference["requested_prompt"])
                self.assertEqual(inference["language"], "it")
                self.assertEqual(inference["model"], "qwen2-audio-7b")
                self.assertEqual(inference["temperature"], 0.0)
                self.assertEqual(inference["seed"], 42)
                self.assertEqual(inference["response_mode"], "json")

    def test_size_rejections_keep_original_batch_result_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = [
                root / "input" / name
                for name in ("too-large-first.wav", "small.wav", "too-large-last.wav")
            ]
            sources[0].parent.mkdir()
            contents = (b"too big", b"ok", b"also too big")
            for source, content in zip(sources, contents, strict=True):
                source.write_bytes(content)
            pipeline = _quiet(
                _config(root, max_file_size=3), FakeProcessor(chunk_count=1), FakeBackend()
            )
            results = pipeline.run(sources)
            self.assertEqual(
                [result["status"] for result in results], ["failed", "completed", "failed"]
            )
            self.assertEqual(
                [Path(result["source"]).name for result in results], [s.name for s in sources]
            )

    def test_force_reprepares_and_retranscribes_completed_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "force.wav"
            source.parent.mkdir()
            source.write_bytes(b"audio")
            processor = FakeProcessor(chunk_count=1)
            backend = FakeBackend()
            self.assertEqual(
                _quiet(_config(root), processor, backend).run()[0]["status"], "completed"
            )
            forced = _quiet(_config(root, force=True), processor, backend)
            self.assertEqual(forced.run()[0]["status"], "completed")
            self.assertEqual(processor.prepare_calls, 2)
            self.assertEqual(backend.calls, [0, 0])

    def test_completed_job_with_missing_artifact_is_reexported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "missing-artifact.wav"
            source.parent.mkdir()
            source.write_bytes(b"audio")
            backend = FakeBackend()
            pipeline = _quiet(_config(root), FakeProcessor(chunk_count=1), backend)
            self.assertEqual(pipeline.run()[0]["status"], "completed")
            metadata = next((root / "process").glob("*/*/metadata.json"))
            job_id = f"{metadata.parent.parent.name}/{metadata.parent.name}"
            (root / "output" / job_id / "transcript.vtt").unlink()
            self.assertEqual(pipeline.run()[0]["status"], "completed")
            self.assertTrue((root / "output" / job_id / "transcript.vtt").is_file())
            self.assertEqual(backend.calls, [0])


class ResumeTests(unittest.TestCase):
    def test_failed_chunk_resumes_checkpoint_and_batch_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = [root / "input" / "first.wav", root / "input" / "second.wav"]
            sources[0].parent.mkdir()
            for source in sources:
                source.write_bytes(source.name.encode())
            config = _config(root, retries=0)
            processor = FakeProcessor()
            failed_backend = FakeBackend(failures={1})
            results = _quiet(config, processor, failed_backend).run(sources)
            self.assertEqual([result["status"] for result in results], ["failed", "failed"])
            self.assertEqual(failed_backend.calls, [0, 1, 0, 1])

            resumed_backend = FakeBackend()
            resumed = _quiet(config, processor, resumed_backend)
            self.assertEqual(
                [result["status"] for result in resumed.run(sources)], ["completed", "completed"]
            )
            self.assertEqual(resumed_backend.calls, [1, 1])

    def test_failed_chunk_leaves_intermediate_partial_and_resume_completes_final_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "intermediate.wav"
            source.parent.mkdir()
            source.write_bytes(b"audio fixture")
            config = _config(root, retries=0)
            processor = FakeProcessor(chunk_count=2)
            failed = _quiet(config, processor, FakeBackend(failures={1}))
            result = failed.run([source])[0]
            self.assertEqual(result["status"], "failed")
            job_id = result["job_id"]
            intermediate = root / "output" / job_id / "intermediate"
            self.assertEqual(
                (intermediate / "transcript.txt").read_text(encoding="utf-8"), "chunk 0\n"
            )
            payload = json.loads((intermediate / "transcript.json").read_text(encoding="utf-8"))
            self.assertIn(
                "Partial intermediate transcript; job is not complete.", payload["warnings"]
            )
            report = json.loads((intermediate / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_chunks"], 1)
            self.assertEqual(report["total_chunks"], 2)
            self.assertIn("chunk 1", report["error"])
            self.assertFalse((root / "output" / job_id / "transcript.json").exists())

            resumed = _quiet(config, processor, FakeBackend())
            self.assertEqual(resumed.run([source])[0]["status"], "completed")
            final_report = json.loads(
                (root / "output" / job_id / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final_report["status"], "mock_completed")
            self.assertEqual(
                (root / "output" / job_id / "transcript.txt").read_text(encoding="utf-8"),
                "chunk 0\nchunk 1\n",
            )

    def test_batch_continues_after_one_file_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = [root / "input" / "first.wav", root / "input" / "second.wav"]
            sources[0].parent.mkdir()
            for source in sources:
                source.write_bytes(source.name.encode())

            class FailOnceBackend(FakeBackend):
                def transcribe(self, chunk, *, language, temperature=None):
                    self.calls.append(chunk.index)
                    if len(self.calls) == 1:
                        raise RuntimeError("first source fails once")
                    return [Segment(chunk.start, chunk.end, "valid transcript")]

            pipeline = _quiet(
                _config(root, retries=0), FakeProcessor(chunk_count=1), FailOnceBackend()
            )
            self.assertEqual(
                [result["status"] for result in pipeline.run(sources)], ["failed", "completed"]
            )

    def test_corrupt_checkpoint_is_discarded_and_source_change_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "corrupt-state.wav"
            source.parent.mkdir()
            source.write_bytes(b"original")
            config = _config(root, retries=0)
            processor = FakeProcessor()
            pipeline = _quiet(config, processor, FakeBackend(failures={1}))
            self.assertEqual(pipeline.run()[0]["status"], "failed")
            found = next((root / "process").glob("*/*/metadata.json")).parent
            job_id = f"{found.parent.name}/{found.name}"
            checkpoint_path = root / "process" / job_id / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["chunks"]["0"][0]["start"] = float("nan")
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            resumed_backend = FakeBackend()
            resumed = _quiet(config, processor, resumed_backend)
            self.assertEqual(resumed.run()[0]["status"], "completed")
            self.assertEqual(resumed_backend.calls, [0, 1])

            changed_source = root / "input" / "changes.wav"
            changed_source.write_bytes(b"before")

            class MutatingProcessor(FakeProcessor):
                def prepare(self, *args, **kwargs):
                    prepared = super().prepare(*args, **kwargs)
                    changed_source.write_bytes(b"changed while preparing")
                    return prepared

            changed_pipeline = _quiet(
                _config(root), MutatingProcessor(chunk_count=1), FakeBackend()
            )
            result = changed_pipeline.run([changed_source])[0]
            self.assertEqual(result["status"], "failed")
            self.assertIn("changed while audio preparation", result["error"])


class _FlakyWatchPipeline:
    """Reports ``first_status`` once, then completes."""

    def __init__(self, source, first_status):
        self.source = source
        self.first_status = first_status
        self.config = type("Config", (), {"retries": 1})()
        self.calls = 0

    def discover(self):
        return [self.source]

    def run(self, sources, *, close_backend):
        self.calls += 1
        status = self.first_status if self.calls == 1 else "completed"
        return [{"status": status, "source": str(self.source)}]


class LockAndWatchTests(unittest.TestCase):
    def test_process_lock_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.lock"
            with ProcessLock(path):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with ProcessLock(path):
                        self.fail("second lock unexpectedly succeeded")

    @staticmethod
    def _run_watch(pipeline, stop_after, fake_stat):
        stop_checks = 0

        def stop():
            nonlocal stop_checks
            stop_checks += 1
            return stop_checks > stop_after

        with patch.object(pipeline_module.time, "sleep", lambda _: None):
            with patch.object(Path, "stat", fake_stat):
                run_watch(pipeline, interval=0.01, stable_scans=2, stop=stop)

    def test_watch_waits_for_stable_file(self):
        source = Path("sample.wav")

        class FakeWatchPipeline:
            calls = []

            def discover(self):
                return [source]

            def run(self, sources, *, close_backend):
                self.calls.append((list(sources), close_backend))
                return [{"status": "completed"} for _ in sources]

        pipeline = FakeWatchPipeline()
        fake_stat = type("Stat", (), {"st_size": 10, "st_mtime_ns": 20})()
        self._run_watch(pipeline, 3, lambda self: fake_stat)
        self.assertEqual(pipeline.calls, [([source], False)])

    def test_watch_retries_failed_and_incomplete_sources(self):
        source = Path("flaky.wav")
        fake_stat = type("Stat", (), {"st_size": 10, "st_mtime_ns": 20})()
        for first_status in ("failed", "incomplete"):
            pipeline = _FlakyWatchPipeline(source, first_status)
            with self.subTest(first_status=first_status):
                self._run_watch(pipeline, 3, lambda self, stat=fake_stat: stat)
                self.assertEqual(pipeline.calls, 2)

    def test_watch_gives_up_after_repeated_incomplete_results(self):
        source = Path("stuck.wav")

        class FakeWatchPipeline:
            def __init__(self):
                self.config = type("Config", (), {"retries": 1})()
                self.calls = 0

            def discover(self):
                return [source]

            def run(self, sources, *, close_backend):
                self.calls += 1
                return [{"status": "incomplete", "source": str(source)}]

        pipeline = FakeWatchPipeline()
        fake_stat = type("Stat", (), {"st_size": 10, "st_mtime_ns": 20})()
        self._run_watch(pipeline, 8, lambda self: fake_stat)
        self.assertEqual(pipeline.calls, 2)

    def test_watch_marks_the_preprocessing_signature_completed(self):
        source = Path("changing.wav")
        file_size = [10]

        class FakeWatchPipeline:
            def __init__(self):
                self.calls = 0

            def discover(self):
                return [source]

            def run(self, sources, *, close_backend):
                self.calls += 1
                if self.calls == 1:
                    file_size[0] = 20
                return [{"status": "completed", "source": str(source)}]

        pipeline = FakeWatchPipeline()

        def fake_stat(_path):
            return type("Stat", (), {"st_size": file_size[0], "st_mtime_ns": 20})()

        self._run_watch(pipeline, 4, fake_stat)
        self.assertEqual(pipeline.calls, 2)


if __name__ == "__main__":
    unittest.main()
