"""Per-chunk recovery ladder, incomplete jobs, zero-speech jobs and job identity."""

import json
import tempfile
import unittest
from pathlib import Path

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application.pipeline import TranscriptionPipeline, _valid_checkpoint
from audio_transcript.domain.models import (
    AudioChunk,
    DegenerateOutputError,
    Segment,
    TranscriptionError,
    TransientBackendError,
)

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeBackend, FakeProcessor, make_config


def _pipeline(config, processor, backend):
    return TranscriptionPipeline(config, processor, backend, FileExporter(), status=lambda _: None)


def _ok(chunk, temperature=None):
    return [Segment(chunk.start, chunk.end, f"chunk {chunk.index}{chunk.part}")]


def _loop(_message="loop"):
    raise DegenerateOutputError(_message)


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.source = self.root / "input" / "sample.wav"
        self.source.parent.mkdir()
        self.source.write_bytes(b"original media")

    def read(self, *parts):
        return json.loads(self.root.joinpath(*parts).read_text(encoding="utf-8"))

    def job_id(self):
        found = next((self.root / "process").glob("*/*/metadata.json")).parent
        return f"{found.parent.name}/{found.name}"


class LadderTests(WorkspaceCase):
    def test_degenerate_output_walks_temperatures_in_order_and_stops_on_success(self):
        def behavior(chunk, temperature):
            if temperature in (None, 0.2):
                raise DegenerateOutputError("loop")
            return _ok(chunk)

        backend = FakeBackend(behavior=behavior)
        processor = FakeProcessor(chunk_count=1)
        config = make_config(self.root, retries=2, fallback_temperatures=(0.2, 0.4, 0.8))
        result = _pipeline(config, processor, backend).run([self.source])[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(backend.attempts, [(0, "", None), (0, "", 0.2), (0, "", 0.4)])
        self.assertEqual(processor.split_calls, [])
        checkpoint = self.read("process", result["job_id"], "checkpoint.json")
        history = checkpoint["attempts"]["0"]
        self.assertEqual(
            [(e["stage"], e["temperature"], e["outcome"]) for e in history],
            [
                ("base", 0.0, "DegenerateOutputError"),
                ("temperature", 0.2, "DegenerateOutputError"),
                ("temperature", 0.4, "ok"),
            ],
        )

    def test_transient_errors_retry_identically_but_degenerate_output_does_not(self):
        script = [TransientBackendError("down"), TransientBackendError("down")]

        def behavior(chunk, temperature):
            if script:
                raise script.pop(0)
            return _ok(chunk)

        backend = FakeBackend(behavior=behavior)
        config = make_config(self.root, retries=2)
        result = _pipeline(config, FakeProcessor(chunk_count=1), backend).run([self.source])[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(backend.attempts, [(0, "", None)] * 3)

        other = self.root / "input" / "other.wav"
        other.write_bytes(b"other media")
        degenerate = FakeBackend(behavior=lambda chunk, temperature: _loop())
        config = make_config(self.root, retries=2, fallback_temperatures=(), split_on_failure=False)
        result = _pipeline(config, FakeProcessor(chunk_count=1), degenerate).run([other])[0]
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(degenerate.attempts), 1)

    def test_exhausted_transient_errors_fail_the_job_instead_of_skipping_chunks(self):
        def behavior(chunk, temperature):
            raise TransientBackendError("connection refused")

        backend = FakeBackend(behavior=behavior)
        config = make_config(self.root, retries=1)
        result = _pipeline(config, FakeProcessor(chunk_count=3), backend).run([self.source])[0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(backend.calls, [0, 0])

    def test_other_transcription_errors_fail_without_the_ladder(self):
        def behavior(chunk, temperature):
            raise TranscriptionError("HTTP 400 audio unsupported")

        backend = FakeBackend(behavior=behavior)
        result = _pipeline(
            make_config(self.root, retries=2), FakeProcessor(chunk_count=2), backend
        ).run([self.source])[0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(backend.attempts), 1)

    def test_split_runs_the_ladder_per_part_and_concatenates_in_order(self):
        def behavior(chunk, temperature):
            if chunk.part == "":
                raise DegenerateOutputError("loop")
            if chunk.part == "b" and temperature is None:
                raise DegenerateOutputError("loop")
            return _ok(chunk)

        backend = FakeBackend(behavior=behavior)
        processor = FakeProcessor(chunk_count=2)
        config = make_config(self.root, fallback_temperatures=(0.3,))
        result = _pipeline(config, processor, backend).run([self.source])[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual([c.index for c in processor.split_calls], [0, 1])
        self.assertEqual(
            backend.attempts[:5],
            [(0, "", None), (0, "", 0.3), (0, "a", None), (0, "b", None), (0, "b", 0.3)],
        )
        text = (self.root / "output" / result["job_id"] / "transcript.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(text, "chunk 0a\nchunk 0b\nchunk 1a\nchunk 1b\n")
        report = self.read("output", result["job_id"], "report.json")
        self.assertEqual(report["execution"]["splits_used"], 2)
        self.assertEqual(report["execution"]["temperature_fallbacks_used"], 4)
        checkpoint = self.read("process", result["job_id"], "checkpoint.json")
        split = [e for e in checkpoint["attempts"]["0"] if e.get("action") == "split"]
        self.assertEqual(split[0]["parts"], ["a", "b"])
        self.assertEqual(split[0]["outcome"], "ok")

    def test_split_is_skipped_when_disabled_and_failure_is_recorded_when_split_raises(self):
        backend = FakeBackend(behavior=lambda chunk, temperature: _loop())
        processor = FakeProcessor(chunk_count=1)
        config = make_config(self.root, split_on_failure=False)
        result = _pipeline(config, processor, backend).run([self.source])[0]
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(processor.split_calls, [])

        other = self.root / "input" / "other.wav"
        other.write_bytes(b"other media")
        processor = FakeProcessor(chunk_count=1)
        processor.split_error = "chunk too short to split"
        result = _pipeline(make_config(self.root), processor, backend).run([other])[0]
        self.assertEqual(result["status"], "incomplete")
        checkpoint = self.read("process", result["job_id"], "checkpoint.json")
        failed = checkpoint["failed"]["0"]
        self.assertIn("too short", failed["error"])
        self.assertEqual(failed["attempts"][-1]["action"], "split")
        self.assertNotEqual(failed["attempts"][-1]["outcome"], "ok")

    def test_context_echo_is_retried_once_without_the_context(self):
        class EchoBackend(FakeBackend):
            def transcribe(self, chunk, *, language, temperature=None, use_prompt=True):
                super().transcribe(
                    chunk, language=language, temperature=temperature, use_prompt=use_prompt
                )
                if use_prompt:
                    raise DegenerateOutputError(
                        "Transcript of audio chunk 0 repeats the instruction prompt instead "
                        "of transcribing speech."
                    )
                return []

        backend = EchoBackend()
        config = make_config(self.root, prompt="Names: Rossi, Bianchi")
        result = _pipeline(config, FakeProcessor(chunk_count=1), backend).run([self.source])[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(backend.prompt_flags, [True, True, True, False])
        history = self.read("process", result["job_id"], "checkpoint.json")["attempts"]["0"]
        self.assertEqual(
            [(entry.get("stage"), entry.get("context")) for entry in history],
            [
                ("base", None),
                ("temperature", None),
                ("temperature", None),
                ("no_context", "omitted"),
            ],
        )
        self.assertEqual(history[-1]["outcome"], "no_speech")

    def test_no_context_rung_needs_an_echo_a_prompt_and_the_setting(self):
        echo = (
            "Transcript of audio chunk 0 repeats the instruction prompt instead of "
            "transcribing speech."
        )
        cases = (
            ({"prompt": "Names: Rossi"}, "loop"),  # other degenerate reason
            ({}, echo),  # no configured prompt
            ({"prompt": "Names: Rossi", "context_free_fallback": False}, echo),
        )
        for index, (overrides, message) in enumerate(cases):
            with self.subTest(overrides=overrides, message=message[:20]):
                source = self.root / "input" / f"case{index}.wav"
                source.write_bytes(f"media {index}".encode())

                def behavior(chunk, temperature, message=message):
                    raise DegenerateOutputError(message)

                backend = FakeBackend(behavior=behavior)
                config = make_config(self.root, split_on_failure=False, **overrides)
                result = _pipeline(config, FakeProcessor(chunk_count=1), backend).run([source])[0]
                self.assertEqual(result["status"], "incomplete")
                self.assertNotIn(False, backend.prompt_flags)

    def test_disabling_the_no_context_rung_changes_job_identity_only_when_off(self):
        def fingerprint(**overrides):
            config = make_config(self.root, **overrides)
            return TranscriptionPipeline(config, FakeProcessor())._fingerprint()

        self.assertEqual(fingerprint(), fingerprint(context_free_fallback=True))
        self.assertNotEqual(fingerprint(), fingerprint(context_free_fallback=False))

    def test_explicit_no_speech_is_a_completed_chunk(self):
        backend = FakeBackend(behavior=lambda chunk, temperature: [])
        result = _pipeline(make_config(self.root), FakeProcessor(chunk_count=2), backend).run(
            [self.source]
        )[0]
        self.assertEqual(result["status"], "completed")
        report = self.read("output", result["job_id"], "report.json")
        self.assertEqual(report["segment_count"], 0)
        self.assertEqual(report["execution"]["chunks_no_speech"], 2)
        self.assertEqual(report["execution"]["chunks_ok"], 0)


class IncompleteJobTests(WorkspaceCase):
    def make_backend(self, bad=1):
        def behavior(chunk, temperature):
            if chunk.index == bad:
                raise DegenerateOutputError("loop")
            return _ok(chunk)

        backend = FakeBackend(behavior=behavior)
        backend.name = "llamacpp"
        return backend

    def test_unrecoverable_chunk_makes_the_job_incomplete_and_a_rerun_finishes_it(self):
        config = make_config(self.root, backend="llamacpp", model="qwen2-audio-7b")
        processor = FakeProcessor(chunk_count=3)
        backend = self.make_backend()
        pipeline = _pipeline(config, processor, backend)
        result = pipeline.run([self.source])[0]
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failed_chunks"], [1])
        job_id = result["job_id"]
        # The ladder ran for the failed chunk and the job moved on to the next chunk.
        self.assertIn(2, backend.calls)
        output = self.root / "output" / job_id
        self.assertFalse((output / "transcript.json").exists())
        self.assertFalse((output / "report.json").exists())
        report = self.read("output", job_id, "intermediate", "report.json")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["failed_chunks"][0]["index"], 1)
        self.assertEqual(
            (report["failed_chunks"][0]["start"], report["failed_chunks"][0]["end"]), (1.0, 2.0)
        )
        self.assertEqual(report["completed_chunks"], 2)
        self.assertEqual(report["execution"]["chunks_failed"], 1)
        self.assertIn("1", report["chunk_attempts"])
        transcript = self.read("output", job_id, "intermediate", "transcript.json")
        self.assertTrue(any(w.startswith("Gap: chunk 1") for w in transcript["warnings"]))
        self.assertEqual([s["text"] for s in transcript["segments"]], ["chunk 0", "chunk 2"])
        metadata = self.read("process", job_id, "metadata.json")
        self.assertEqual(metadata["status"], "incomplete")
        self.assertEqual(metadata["failed_chunks"], [1])
        self.assertTrue(self.source.exists())
        self.assertFalse((self.root / "processed" / job_id.split("/")[0]).exists())

        good = FakeBackend()
        good.name = "llamacpp"
        resumed = _pipeline(config, processor, good).run([self.source])[0]
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(good.calls, [1])
        self.assertEqual(processor.prepare_calls, 1)
        text = (output / "transcript.txt").read_text(encoding="utf-8")
        self.assertEqual(text, "chunk 0\nchunk 1\nchunk 2\n")
        self.assertFalse(self.source.exists())
        self.assertTrue((self.root / "processed" / job_id.split("/")[0] / "sample.wav").is_file())
        final_report = self.read("output", job_id, "intermediate", "report.json")
        self.assertEqual(final_report["status"], "completed")
        self.assertEqual(final_report["failed_chunks"], [])

    def test_rerun_retries_failed_chunks_that_fail_again(self):
        config = make_config(self.root, backend="llamacpp", model="qwen2-audio-7b")
        processor = FakeProcessor(chunk_count=2)
        _pipeline(config, processor, self.make_backend(bad=0)).run([self.source])
        backend = self.make_backend(bad=0)
        result = _pipeline(config, processor, backend).run([self.source])[0]
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual({index for index in backend.calls}, {0})

    def test_checkpoint_validation_covers_failed_and_attempt_keys(self):
        chunks = [AudioChunk(Path("a.wav"), 0.0, 1.0, 0), AudioChunk(Path("b.wav"), 1.0, 2.0, 1)]
        valid = {
            "fingerprint": "f",
            "chunks": {"0": []},
            "failed": {"1": {"start": 1.0, "end": 2.0, "error": "x", "attempts": []}},
            "attempts": {"0": [{"outcome": "ok"}]},
        }
        self.assertTrue(_valid_checkpoint(valid, "f", chunks))
        variants = {
            "failed not dict": {**valid, "failed": []},
            "unknown failed index": {**valid, "failed": {"9": valid["failed"]["1"]}},
            "failed and done": {**valid, "failed": {"0": valid["failed"]["1"]}},
            "failed missing error": {**valid, "failed": {"1": {"start": 1.0, "end": 2.0}}},
            "attempts not list": {**valid, "attempts": {"0": "bad"}},
            "attempts unknown index": {**valid, "attempts": {"7": []}},
        }
        for label, value in variants.items():
            with self.subTest(label):
                self.assertFalse(_valid_checkpoint(value, "f", chunks))


class ZeroSpeechAndArchiveTests(WorkspaceCase):
    def test_zero_speech_job_completes_with_warning_and_no_segments(self):
        backend = FakeBackend()
        processor = FakeProcessor(chunk_count=4, empty=True)
        config = make_config(self.root)
        pipeline = _pipeline(config, processor, backend)
        result = pipeline.run([self.source])[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(backend.calls, [])
        transcript = self.read("output", result["job_id"], "transcript.json")
        self.assertEqual(transcript["segments"], [])
        self.assertIn("No speech detected", transcript["warnings"])
        self.assertEqual(transcript["duration"], 4.0)
        report = self.read("output", result["job_id"], "report.json")
        self.assertEqual(report["segment_count"], 0)
        self.assertEqual(report["audio_metadata"]["speech_seconds"], 0.0)
        self.assertEqual(report["audio_metadata"]["chunk_count"], 0)
        self.assertEqual(pipeline.run([self.source])[0]["status"], "skipped")
        prepared = self.read("process", result["job_id"], "prepared.json")
        self.assertEqual(prepared["chunks"], [])
        self.assertEqual(prepared["duration"], 4.0)

    def test_prepared_manifest_and_audio_metadata_use_the_full_duration(self):
        processor = FakeProcessor(chunk_count=2, trailing_silence=6.0)
        config = make_config(self.root, chunk_seconds=12.0, vad_threshold=0.6)
        result = _pipeline(config, processor, FakeBackend()).run([self.source])[0]
        prepared = self.read("process", result["job_id"], "prepared.json")
        self.assertEqual(prepared["duration"], 8.0)
        self.assertEqual(prepared["speech_seconds"], 2.0)
        self.assertEqual(prepared["segmentation"]["detector"], "none")
        self.assertTrue(all(c["part"] == "" for c in prepared["chunks"]))
        metadata = self.read("process", result["job_id"], "metadata.json")
        audio = metadata["audio_metadata"]
        self.assertEqual(audio["duration_seconds"], 8.0)
        self.assertEqual(audio["speech_seconds"], 2.0)
        self.assertEqual(audio["chunk_count"], 2)
        self.assertEqual(audio["segmentation"], prepared["segmentation"])
        settings = processor.prepare_kwargs[0]["segmentation"]
        self.assertEqual((settings.detector, settings.max_chunk_seconds), ("none", 12.0))
        self.assertEqual(settings.threshold, 0.6)

    def test_trailing_silence_does_not_block_archiving(self):
        backend = FakeBackend()
        backend.name = "llamacpp"
        processor = FakeProcessor(chunk_count=1, trailing_silence=30.0)
        config = make_config(self.root, backend="llamacpp", model="qwen2-audio-7b")
        result = _pipeline(config, processor, backend).run([self.source])[0]
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("archive_warning", result)
        self.assertFalse(self.source.exists())
        self.assertTrue(
            (self.root / "processed" / result["job_id"].split("/")[0] / "sample.wav").is_file()
        )

    def test_zero_speech_real_backend_job_archives_the_fully_analysed_input(self):
        backend = FakeBackend()
        backend.name = "llamacpp"
        config = make_config(self.root, backend="llamacpp", model="qwen2-audio-7b")
        result = _pipeline(config, FakeProcessor(chunk_count=2, empty=True), backend).run(
            [self.source]
        )[0]
        self.assertEqual(result["status"], "completed")
        self.assertIn("archived_source_path", result)


class FingerprintTests(WorkspaceCase):
    def fingerprint(self, **overrides):
        config = make_config(self.root, backend="llamacpp", model="qwen2-audio-7b", **overrides)
        return _pipeline(config, FakeProcessor(), None)._fingerprint()

    def test_vad_fallback_and_generation_settings_change_job_identity(self):
        base = self.fingerprint()
        self.assertEqual(base, self.fingerprint())
        changes = [
            {"vad": "energy"},
            {"vad_threshold": 0.6},
            {"vad_min_speech_seconds": 0.5},
            {"vad_min_silence_seconds": 0.3},
            {"vad_speech_pad_seconds": 0.1},
            {"vad_max_merge_gap_seconds": 2.0},
            {"vad_energy_margin_db": 10.0},
            {"chunk_seconds": 12.0},
            {"fallback_temperatures": (0.3,)},
            {"split_on_failure": False},
            {"min_tokens": 64},
            {"tokens_per_second": 6.0},
            {"compression_ratio_threshold": 2.0},
            {"repeat_penalty": 1.1},
            {"dry_multiplier": 0.8},
            {"response_mode": "qwen3-asr"},
        ]
        seen = {base}
        for change in changes:
            with self.subTest(change=change):
                value = self.fingerprint(**change)
                self.assertNotIn(value, seen)
                seen.add(value)

    def test_vad_model_file_signature_is_part_of_the_fingerprint(self):
        model = self.root / "vad.onnx"
        model.write_bytes(b"first model")
        first = self.fingerprint(vad="silero", vad_model_path=model)
        model.write_bytes(b"a different, longer model")
        self.assertNotEqual(first, self.fingerprint(vad="silero", vad_model_path=model))
        # The model file is irrelevant when another detector is selected.
        model.write_bytes(b"x")
        energy = self.fingerprint(vad="energy", vad_model_path=model)
        model.write_bytes(b"yy")
        self.assertEqual(energy, self.fingerprint(vad="energy", vad_model_path=model))


class StatusLineTests(WorkspaceCase):
    def test_run_log_lines_are_informative_and_free_of_transcript_text(self):
        class MetricBackend(FakeBackend):
            metrics = None

            def transcribe(self, chunk, *, language, temperature=None):
                if chunk.index == 1 and temperature is None:
                    self.metrics = {"completion_tokens": 168, "finish_reason": "length"}
                    raise DegenerateOutputError("hit the token limit")
                self.metrics = {"completion_tokens": 37, "finish_reason": "stop"}
                return [Segment(chunk.start, chunk.end, "secret words")]

            def last_call_metrics(self):
                return self.metrics

        lines = []
        config = make_config(self.root, fallback_temperatures=(0.2,))
        pipeline = TranscriptionPipeline(
            config,
            FakeProcessor(chunk_count=3),
            MetricBackend(),
            FileExporter(),
            status=lines.append,
        )
        self.assertEqual(pipeline.run([self.source])[0]["status"], "completed")
        text = "\n".join(lines)
        self.assertNotIn("secret words", text)
        self.assertRegex(lines[0], r"^Prepared sample\.wav: 3\.0s analysed, detector=none, speech")
        self.assertIn("3 chunks, length min/avg/max 1.0/1.0/1.0s", lines[0])
        self.assertIn("Chunk 1/3 [0.0-1.0s, 1.0s audio] t=0 ok", text)
        self.assertRegex(
            text,
            r"Chunk 2/3 \[1\.0-2\.0s, 1\.0s audio\] t=0 DegenerateOutputError\(length\) "
            r"[\d.]+s 168 tok -> fallback t=0\.2",
        )
        self.assertRegex(text, r"Chunk 2/3 .* t=0\.2 ok [\d.]+s 37 tok")
        execution = next(line for line in lines if line.startswith("Execution:"))
        self.assertIn("ok=3 no_speech=0 failed=0", execution)
        self.assertIn("fallbacks=1 splits=0", execution)
        self.assertTrue(lines[-2].startswith("Completed sample.wav"))
        self.assertTrue(lines[-1].startswith("Timings: total "))
        self.assertIn("| preparation ", lines[-1])
        self.assertIn("| inference ", lines[-1])

    def test_failed_ladder_and_split_lines(self):
        lines = []
        pipeline = TranscriptionPipeline(
            make_config(self.root, fallback_temperatures=()),
            FakeProcessor(chunk_count=1),
            FakeBackend(behavior=lambda chunk, temperature: _loop()),
            FileExporter(),
            status=lines.append,
        )
        self.assertEqual(pipeline.run([self.source])[0]["status"], "incomplete")
        text = "\n".join(lines)
        self.assertIn("DegenerateOutputError(invalid)", text)
        self.assertIn("-> split", text)
        self.assertIn("Chunk 1/1 [0.0-1.0s, 1.0s audio] split a/b", text)
        self.assertIn("Chunk 1a/1", text)
        self.assertIn("FAILED after ladder -> continuing", text)
        self.assertTrue(lines[-2].startswith("Incomplete sample.wav"))
        self.assertTrue(lines[-1].startswith("Timings: total "))


class ExecutionMetricsTests(WorkspaceCase):
    def test_execution_summary_aggregates_backend_metrics(self):
        class MetricBackend(FakeBackend):
            def __init__(self):
                super().__init__()
                self.metrics = None

            def transcribe(self, chunk, *, language, temperature=None):
                self.metrics = None
                if chunk.index == 1 and temperature is None:
                    self.metrics = {"completion_tokens": 48, "finish_reason": "length"}
                    raise DegenerateOutputError("loop")
                tokens = {0: 10, 1: 30, 2: 20}[chunk.index]
                self.metrics = {"completion_tokens": tokens, "finish_reason": "stop"}
                return _ok(chunk)

            def last_call_metrics(self):
                return self.metrics

        config = make_config(self.root, fallback_temperatures=(0.2,))
        result = _pipeline(config, FakeProcessor(chunk_count=3), MetricBackend()).run(
            [self.source]
        )[0]
        report = self.read("output", result["job_id"], "report.json")
        execution = report["execution"]
        self.assertEqual(execution["schema_version"], 3)
        self.assertEqual(execution["total_chunks"], 3)
        self.assertEqual(
            (execution["chunks_ok"], execution["chunks_no_speech"], execution["chunks_failed"]),
            (3, 0, 0),
        )
        self.assertEqual(execution["attempts"], 4)
        self.assertEqual(execution["temperature_fallbacks_used"], 1)
        self.assertEqual(execution["splits_used"], 0)
        self.assertEqual(execution["audio_seconds_sent"], 4.0)
        self.assertEqual(execution["analysed_audio_seconds"], 3.0)
        self.assertGreaterEqual(execution["inference_wall_seconds"], 0)
        self.assertIsNotNone(execution["real_time_factor"])
        tokens = execution["completion_tokens"]
        self.assertEqual((tokens["sum"], tokens["max"], tokens["p95"]), (108, 48, 48))
        checkpoint = self.read("process", result["job_id"], "checkpoint.json")
        first = checkpoint["attempts"]["1"][0]
        self.assertEqual(first["metrics"]["finish_reason"], "length")
        self.assertEqual(first["audio_seconds"], 1.0)
        self.assertEqual(first["part"], "")
        self.assertIn("wall_seconds", first)
        # Metrics carry numbers only, never transcript or prompt text.
        self.assertNotIn("chunk 0", json.dumps(checkpoint["attempts"]))

    def test_backend_without_metrics_records_null(self):
        result = _pipeline(make_config(self.root), FakeProcessor(chunk_count=1), FakeBackend()).run(
            [self.source]
        )[0]
        checkpoint = self.read("process", result["job_id"], "checkpoint.json")
        self.assertIsNone(checkpoint["attempts"]["0"][0]["metrics"])
        report = self.read("output", result["job_id"], "report.json")
        self.assertIsNone(report["execution"]["completion_tokens"]["max"])


if __name__ == "__main__":
    unittest.main()
