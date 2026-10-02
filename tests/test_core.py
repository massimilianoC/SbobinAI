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
from audio_transcript.config import AppConfig, load_config
from audio_transcript.domain.models import AudioChunk, MediaInfo, Segment


def _config(root: Path, **overrides) -> AppConfig:
    options = {
        "input_dir": root / "input",
        "processed_dir": root / "processed",
        "process_dir": root / "process",
        "output_dir": root / "output",
        "backend": "mock",
        "model": "mock",
    }
    options.update(overrides)
    return AppConfig(**options)


class FakeProcessor:
    def __init__(self, chunk_count=2):
        self.chunk_count = chunk_count
        self.prepare_calls = 0

    def probe(self, source):
        return MediaInfo(
            duration=float(self.chunk_count), sample_rate=16000, channels=1, codec="fake"
        )

    def prepare(self, source, destination, *, chunk_seconds, sample_rate, max_duration=None):
        self.prepare_calls += 1
        chunks = []
        for index in range(self.chunk_count):
            path = destination / f"chunk_{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic test fixture")
            chunks.append(
                AudioChunk(path=path, start=float(index), end=float(index + 1), index=index)
            )
        return chunks


class FakeBackend:
    name = "mock"

    def __init__(self, failures=()):
        self.failures = set(failures)
        self.calls = []

    def check(self):
        return None

    def transcribe(self, chunk, *, language):
        self.calls.append(chunk.index)
        if chunk.index in self.failures:
            raise RuntimeError("synthetic backend failure")
        return [Segment(chunk.start, chunk.end, f"chunk {chunk.index}")]

    def close(self):
        return None


def test_config_overrides_toml_and_validates_types():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_file = root / "settings.toml"
        config_file.write_text('backend = "mock"\nchunk_seconds = 12\n', encoding="utf-8")
        config = load_config(config_file, {"chunk_seconds": 8.0})
        assert config.backend == "mock"
        assert config.model == "mock"
        assert config.chunk_seconds == 8

        config_file.write_text('sample_rate = "16000"\n', encoding="utf-8")
        try:
            load_config(config_file)
        except ValueError as exc:
            assert "sample_rate" in str(exc)
        else:
            raise AssertionError("invalid TOML type must fail with a configuration error")


def test_llamacpp_config_defaults_and_validation():
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
    assert config.model == "qwen2-audio-7b"
    assert config.base_url == "https://localhost:8088/v1"
    assert config.timeout == 45
    assert config.max_tokens == 512
    assert config.temperature == 0.4
    assert config.seed == 123
    assert config.response_mode == "plain"
    assert config.prompt == "Transcribe the speech."
    for invalid in (
        {"backend": "llamacpp", "base_url": "file:///tmp/model"},
        {"backend": "llamacpp", "timeout": 0},
        {"backend": "llamacpp", "max_tokens": 0},
        {"backend": "llamacpp", "max_tokens": True},
        {"backend": "llamacpp", "temperature": 2.1},
        {"backend": "llamacpp", "seed": -1},
        {"backend": "llamacpp", "response_mode": "xml"},
    ):
        try:
            load_config(overrides=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid llama.cpp config should fail: {invalid}")


def test_llamacpp_factory_and_doctor_use_selected_backend():
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
    assert backend.kwargs == {
        "model": "qwen2-audio-7b",
        "base_url": "http://127.0.0.1:8088",
        "timeout": 12,
        "max_tokens": 256,
        "temperature": 0.0,
        "seed": 42,
        "response_mode": "json",
        "prompt": None,
    }

    backend = Backend()
    with patch("audio_transcript.cli._make_backend", return_value=backend):
        with patch("audio_transcript.cli.shutil.which", return_value="ffmpeg"):
            with patch(
                "audio_transcript.cli.subprocess.run",
                return_value=type("Result", (), {"returncode": 0, "stderr": ""})(),
            ):
                assert _doctor(config) == 0
    assert backend.checked


def test_run_skips_completed_and_isolates_configuration_identity():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "café audio.wav"
        source.parent.mkdir()
        source.write_bytes(b"original")
        processor = FakeProcessor(chunk_count=1)
        first_backend = FakeBackend()
        first_config = _config(root)
        first = TranscriptionPipeline(
            first_config, processor, first_backend, FileExporter(), status=lambda _: None
        )
        assert first.run()[0]["status"] == "completed"
        assert source.read_bytes() == b"original"
        assert first.run()[0]["status"] == "skipped"
        assert processor.prepare_calls == 1

        changed_config = _config(root, model="another-model")
        changed = TranscriptionPipeline(
            changed_config, processor, FakeBackend(), FileExporter(), status=lambda _: None
        )
        assert changed.run()[0]["status"] == "completed"
        jobs = list((root / "output").iterdir())
        assert len(jobs) == 2


def test_real_backend_archives_only_direct_queued_input_after_success():
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
        pipeline = TranscriptionPipeline(
            config, FakeProcessor(chunk_count=1), backend, FileExporter(), status=lambda _: None
        )
        result = pipeline.run([source, external])
        assert [item["status"] for item in result] == ["completed", "completed"]
        assert not source.exists()
        assert (root / "processed" / result[0]["job_id"] / source.name).read_bytes() == original
        assert external.read_bytes() == b"external media"
        metadata = json.loads(
            (root / "process" / result[0]["job_id"] / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["archived_source_path"].endswith(source.name)
        assert not any(path.is_dir() for path in (root / "input").iterdir())


def test_archive_disabled_mock_and_prepare_only_preserve_sources():
    for mode in ("mock", "disabled", "prepare-only"):
        with tempfile.TemporaryDirectory() as temporary:
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
                    root,
                    backend="llamacpp",
                    model="qwen2-audio-7b",
                    prepare_only=True,
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
                assert pipeline.run([source])[0]["status"] == "skipped"
            assert source.read_bytes() == b"retain original"


def test_archive_collision_keeps_completed_transcript_and_retries_later():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "collision.wav"
        source.parent.mkdir()
        source.write_bytes(b"original input")
        backend = FakeBackend()
        backend.name = "llamacpp"
        config = _config(root, backend="llamacpp", model="qwen2-audio-7b")
        pipeline = TranscriptionPipeline(
            config, FakeProcessor(chunk_count=1), backend, FileExporter(), status=lambda _: None
        )
        job_id, _ = pipeline._identity(source, pipeline._fingerprint())
        target = root / "processed" / job_id / source.name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"different data")
        result = pipeline.run([source])[0]
        assert result["status"] == "completed"
        assert "archive_warning" in result
        assert source.read_bytes() == b"original input"
        assert target.read_bytes() == b"different data"
        metadata_path = root / "process" / job_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["status"] == "completed"
        assert "archive_warning" in metadata
        assert not (root / "process" / job_id / "failure.json").exists()
        resumed = pipeline.run([source])[0]
        assert resumed["status"] == "skipped"
        assert "archive_warning" in resumed

        target.write_bytes(b"original input")
        resumed = pipeline.run([source])[0]
        assert resumed["status"] == "skipped"
        assert not source.exists()
        assert target.read_bytes() == b"original input"


def test_bounded_real_backend_run_retains_full_source_and_does_not_archive_on_skip():
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
        config = _config(
            root,
            backend="llamacpp",
            model="qwen2-audio-7b",
            max_duration=1.0,
        )
        pipeline = TranscriptionPipeline(
            config,
            BoundedProcessor(chunk_count=1),
            backend,
            FileExporter(),
            status=lambda _: None,
        )
        result = pipeline.run([source])[0]
        assert result["status"] == "completed"
        assert "Only part of the source was processed" in result["archive_warning"]
        assert source.read_bytes() == b"full source media"
        assert not (root / "processed" / result["job_id"] / source.name).exists()

        resumed = pipeline.run([source])[0]
        assert resumed["status"] == "skipped"
        assert "Only part of the source was processed" in resumed["archive_warning"]
        assert source.read_bytes() == b"full source media"


def test_llamacpp_generation_settings_are_recorded_and_isolate_jobs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "settings.wav"
        source.parent.mkdir()
        source.write_bytes(b"original")
        backend = FakeBackend()
        backend.name = "llamacpp"
        first_config = _config(
            root,
            backend="llamacpp",
            model="qwen2-audio-7b",
            archive_inputs=False,
            temperature=0.25,
            seed=17,
            response_mode="plain",
            prompt="Transcribe faithfully.",
        )
        first = TranscriptionPipeline(
            first_config,
            FakeProcessor(chunk_count=1),
            backend,
            FileExporter(),
            status=lambda _: None,
        )
        result = first.run()[0]
        report = json.loads(
            (root / "output" / result["job_id"] / "report.json").read_text(encoding="utf-8")
        )
        assert report["inference_configuration"]["temperature"] == 0.25
        assert report["inference_configuration"]["seed"] == 17
        assert report["inference_configuration"]["response_mode"] == "plain"
        assert report["inference_configuration"]["prompt"] == "Transcribe faithfully."

        second_config = _config(
            root,
            backend="llamacpp",
            model="qwen2-audio-7b",
            archive_inputs=False,
            temperature=0.5,
            seed=17,
            response_mode="plain",
            prompt="Transcribe faithfully.",
        )
        second = TranscriptionPipeline(
            second_config,
            FakeProcessor(chunk_count=1),
            FakeBackend(),
            FileExporter(),
            status=lambda _: None,
        )
        assert second.run([source])[0]["job_id"] != result["job_id"]


def test_failed_chunk_resumes_checkpoint_and_batch_continues():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sources = [root / "input" / "first.wav", root / "input" / "second.wav"]
        sources[0].parent.mkdir()
        for source in sources:
            source.write_bytes(source.name.encode())
        config = _config(root, retries=0)
        processor = FakeProcessor()
        failed_backend = FakeBackend(failures={1})
        failed_pipeline = TranscriptionPipeline(
            config, processor, failed_backend, FileExporter(), status=lambda _: None
        )
        results = failed_pipeline.run(sources)
        assert [result["status"] for result in results] == ["failed", "failed"]
        assert failed_backend.calls == [0, 1, 0, 1]

        resumed_backend = FakeBackend()
        resumed = TranscriptionPipeline(
            config, processor, resumed_backend, FileExporter(), status=lambda _: None
        )
        assert [result["status"] for result in resumed.run(sources)] == ["completed", "completed"]
        assert resumed_backend.calls == [1, 1]


def test_failed_chunk_leaves_intermediate_partial_and_resume_completes_final_exports():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "intermediate.wav"
        source.parent.mkdir()
        source.write_bytes(b"audio fixture")
        config = _config(root, retries=0)
        processor = FakeProcessor(chunk_count=2)
        failed = TranscriptionPipeline(
            config,
            processor,
            FakeBackend(failures={1}),
            FileExporter(),
            status=lambda _: None,
        )
        result = failed.run([source])[0]
        assert result["status"] == "failed"
        job_id = result["job_id"]
        intermediate = root / "output" / job_id / "intermediate"
        partial_text = (intermediate / "transcript.txt").read_text(encoding="utf-8")
        assert partial_text == "chunk 0\n"
        payload = json.loads((intermediate / "transcript.json").read_text(encoding="utf-8"))
        assert "Partial intermediate transcript; job is not complete." in payload["warnings"]
        report = json.loads((intermediate / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "failed"
        assert report["completed_chunks"] == 1
        assert report["total_chunks"] == 2
        assert "chunk 1" in report["error"]
        assert not (root / "output" / job_id / "transcript.json").exists()

        resumed = TranscriptionPipeline(
            config,
            processor,
            FakeBackend(),
            FileExporter(),
            status=lambda _: None,
        )
        assert resumed.run([source])[0]["status"] == "completed"
        final_report = json.loads(
            (root / "output" / job_id / "report.json").read_text(encoding="utf-8")
        )
        assert final_report["status"] == "mock_completed"
        assert (root / "output" / job_id / "transcript.txt").read_text(encoding="utf-8") == (
            "chunk 0\nchunk 1\n"
        )


def test_effective_prompt_is_written_to_transcript_and_report():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "prompt.wav"
        source.parent.mkdir()
        source.write_bytes(b"audio fixture")

        class PromptBackend(FakeBackend):
            name = "llamacpp"

            def effective_prompt(self, language):
                return f"Resolved Italian prompt ({language})"

        backend = PromptBackend()
        config = _config(
            root,
            backend="llamacpp",
            model="qwen2-audio-7b",
            language="it",
            archive_inputs=False,
        )
        pipeline = TranscriptionPipeline(
            config, FakeProcessor(chunk_count=1), backend, FileExporter(), status=lambda _: None
        )
        result = pipeline.run([source])[0]
        job_dir = root / "output" / result["job_id"]
        report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
        transcript = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
        for document in (report, transcript):
            inference = document["inference_configuration"]
            assert inference["prompt"] == "Resolved Italian prompt (it)"
            assert inference["requested_prompt"] is None
            assert inference["language"] == "it"
            assert inference["model"] == "qwen2-audio-7b"
            assert inference["temperature"] == 0.0
            assert inference["seed"] == 42
            assert inference["response_mode"] == "json"


def test_batch_continues_after_one_file_failure():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sources = [root / "input" / "first.wav", root / "input" / "second.wav"]
        sources[0].parent.mkdir()
        for source in sources:
            source.write_bytes(source.name.encode())
        config = _config(root, retries=0)

        class FailOnceBackend(FakeBackend):
            def transcribe(self, chunk, *, language):
                self.calls.append(chunk.index)
                if len(self.calls) == 1:
                    raise RuntimeError("first source fails once")
                return [Segment(chunk.start, chunk.end, "valid transcript")]

        pipeline = TranscriptionPipeline(
            config,
            FakeProcessor(chunk_count=1),
            FailOnceBackend(),
            FileExporter(),
            status=lambda _: None,
        )
        assert [result["status"] for result in pipeline.run(sources)] == ["failed", "completed"]


def test_size_rejections_keep_original_batch_result_order():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sources = [
            root / "input" / name
            for name in ("too-large-first.wav", "small.wav", "too-large-last.wav")
        ]
        sources[0].parent.mkdir()
        for source, content in zip(sources, (b"too big", b"ok", b"also too big"), strict=True):
            source.write_bytes(content)
        config = _config(root, max_file_size=3)
        pipeline = TranscriptionPipeline(
            config,
            FakeProcessor(chunk_count=1),
            FakeBackend(),
            FileExporter(),
            status=lambda _: None,
        )
        results = pipeline.run(sources)
        assert [result["status"] for result in results] == ["failed", "completed", "failed"]
        assert [Path(result["source"]).name for result in results] == [
            source.name for source in sources
        ]


def test_force_reprepares_and_retranscribes_completed_job():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "force.wav"
        source.parent.mkdir()
        source.write_bytes(b"audio")
        processor = FakeProcessor(chunk_count=1)
        config = _config(root)
        backend = FakeBackend()
        pipeline = TranscriptionPipeline(
            config, processor, backend, FileExporter(), status=lambda _: None
        )
        assert pipeline.run()[0]["status"] == "completed"
        forced = TranscriptionPipeline(
            _config(root, force=True), processor, backend, FileExporter(), status=lambda _: None
        )
        assert forced.run()[0]["status"] == "completed"
        assert processor.prepare_calls == 2
        assert backend.calls == [0, 0]


def test_completed_job_with_missing_artifact_is_reexported():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "missing-artifact.wav"
        source.parent.mkdir()
        source.write_bytes(b"audio")
        backend = FakeBackend()
        pipeline = TranscriptionPipeline(
            _config(root),
            FakeProcessor(chunk_count=1),
            backend,
            FileExporter(),
            status=lambda _: None,
        )
        assert pipeline.run()[0]["status"] == "completed"
        metadata = next((root / "process").glob("*/metadata.json"))
        job_id = metadata.parent.name
        (root / "output" / job_id / "transcript.vtt").unlink()
        assert pipeline.run()[0]["status"] == "completed"
        assert (root / "output" / job_id / "transcript.vtt").is_file()
        assert backend.calls == [0]


def test_corrupt_checkpoint_is_discarded_and_source_change_fails():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "input" / "corrupt-state.wav"
        source.parent.mkdir()
        source.write_bytes(b"original")
        config = _config(root, retries=0)
        processor = FakeProcessor()
        backend = FakeBackend(failures={1})
        pipeline = TranscriptionPipeline(
            config, processor, backend, FileExporter(), status=lambda _: None
        )
        assert pipeline.run()[0]["status"] == "failed"
        job_id = next((root / "process").glob("*/metadata.json")).parent.name
        checkpoint_path = root / "process" / job_id / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["chunks"]["0"][0]["start"] = float("nan")
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        resumed_backend = FakeBackend()
        resumed = TranscriptionPipeline(
            config, processor, resumed_backend, FileExporter(), status=lambda _: None
        )
        assert resumed.run()[0]["status"] == "completed"
        assert resumed_backend.calls == [0, 1]

        changed_source = root / "input" / "changes.wav"
        changed_source.write_bytes(b"before")

        class MutatingProcessor(FakeProcessor):
            def prepare(self, *args, **kwargs):
                chunks = super().prepare(*args, **kwargs)
                changed_source.write_bytes(b"changed while preparing")
                return chunks

        changed_pipeline = TranscriptionPipeline(
            _config(root),
            MutatingProcessor(chunk_count=1),
            FakeBackend(),
            FileExporter(),
            status=lambda _: None,
        )
        result = changed_pipeline.run([changed_source])[0]
        assert result["status"] == "failed"
        assert "changed while audio preparation" in result["error"]


def test_process_lock_rejects_a_second_writer():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "run.lock"
        with ProcessLock(path):
            try:
                with ProcessLock(path):
                    raise AssertionError("second lock unexpectedly succeeded")
            except RuntimeError as exc:
                assert "already running" in str(exc)


def test_watch_waits_for_stable_file():
    source = Path("sample.wav")

    class FakeWatchPipeline:
        calls = []

        def discover(self):
            return [source]

        def run(self, sources, *, close_backend):
            self.calls.append((list(sources), close_backend))
            return [{"status": "completed"} for _ in sources]

    pipeline = FakeWatchPipeline()
    stop_checks = 0

    def stop():
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks > 3

    fake_stat = type("Stat", (), {"st_size": 10, "st_mtime_ns": 20})()
    with patch.object(pipeline_module.time, "sleep", lambda _: None):
        with patch.object(Path, "stat", lambda self: fake_stat):
            run_watch(pipeline, interval=0.01, stable_scans=2, stop=stop)
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0] == ([source], False)


def test_watch_retries_failed_source():
    source = Path("flaky.wav")

    class FakeWatchPipeline:
        def __init__(self):
            self.config = type("Config", (), {"retries": 1})()
            self.calls = 0

        def discover(self):
            return [source]

        def run(self, sources, *, close_backend):
            self.calls += 1
            status = "failed" if self.calls == 1 else "completed"
            return [{"status": status, "source": str(source)}]

    pipeline = FakeWatchPipeline()
    stop_checks = 0

    def stop():
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks > 3

    fake_stat = type("Stat", (), {"st_size": 10, "st_mtime_ns": 20})()
    with patch.object(pipeline_module.time, "sleep", lambda _: None):
        with patch.object(Path, "stat", lambda self: fake_stat):
            run_watch(pipeline, interval=0.01, stable_scans=2, stop=stop)
    assert pipeline.calls == 2


def test_watch_marks_the_preprocessing_signature_completed():
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
    stop_checks = 0

    def stop():
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks > 4

    def fake_stat(_path):
        return type("Stat", (), {"st_size": file_size[0], "st_mtime_ns": 20})()

    with patch.object(pipeline_module.time, "sleep", lambda _: None):
        with patch.object(Path, "stat", fake_stat):
            run_watch(pipeline, interval=0.01, stable_scans=2, stop=stop)
    assert pipeline.calls == 2


def load_tests(loader, tests, pattern):
    """Expose the focused checks to unittest without a third-party runner."""
    del loader, tests, pattern
    checks = (
        test_config_overrides_toml_and_validates_types,
        test_llamacpp_config_defaults_and_validation,
        test_llamacpp_factory_and_doctor_use_selected_backend,
        test_real_backend_archives_only_direct_queued_input_after_success,
        test_archive_disabled_mock_and_prepare_only_preserve_sources,
        test_archive_collision_keeps_completed_transcript_and_retries_later,
        test_bounded_real_backend_run_retains_full_source_and_does_not_archive_on_skip,
        test_llamacpp_generation_settings_are_recorded_and_isolate_jobs,
        test_run_skips_completed_and_isolates_configuration_identity,
        test_failed_chunk_resumes_checkpoint_and_batch_continues,
        test_failed_chunk_leaves_intermediate_partial_and_resume_completes_final_exports,
        test_effective_prompt_is_written_to_transcript_and_report,
        test_batch_continues_after_one_file_failure,
        test_size_rejections_keep_original_batch_result_order,
        test_force_reprepares_and_retranscribes_completed_job,
        test_completed_job_with_missing_artifact_is_reexported,
        test_corrupt_checkpoint_is_discarded_and_source_change_fails,
        test_process_lock_rejects_a_second_writer,
        test_watch_waits_for_stable_file,
        test_watch_retries_failed_source,
        test_watch_marks_the_preprocessing_signature_completed,
    )
    return unittest.TestSuite(unittest.FunctionTestCase(check) for check in checks)
