"""Concurrent chunk inference, thread-local metrics, report timings and export throttling."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.adapters.llamacpp import LlamaCppBackend
from audio_transcript.application import pipeline as pipeline_module
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.config import load_config
from audio_transcript.domain.models import (
    AudioChunk,
    DegenerateOutputError,
    Segment,
    TransientBackendError,
)

try:
    from .fakes import FakeProcessor, make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeProcessor, make_config


class SleepingBackend:
    """Thread-safe fake that sleeps per request and records the peak concurrency."""

    name = "mock"

    def __init__(self, delay=0.05, behavior=None):
        self.delay = delay
        self.behavior = behavior
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[tuple[int, str, float | None]] = []

    def check(self):
        return None

    def transcribe(self, chunk, *, language, temperature=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((chunk.index, chunk.part, temperature))
        try:
            if self.behavior is not None:
                outcome = self.behavior(chunk, temperature)
                if outcome is not None:
                    return outcome
            time.sleep(self.delay)
            return [Segment(chunk.start, chunk.end, f"chunk {chunk.index}{chunk.part}")]
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        return None


class Workspace(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.base = Path(self._temporary.name)

    def make_root(self, name="a") -> Path:
        root = self.base / name
        (root / "input").mkdir(parents=True)
        (root / "input" / "sample.wav").write_bytes(b"original media")
        return root

    def run_job(self, root, backend, *, chunks=6, processor=None, **config):
        options = {"archive_inputs": False, "retries": 0}
        options.update(config)
        pipeline = TranscriptionPipeline(
            make_config(root, **options),
            processor or FakeProcessor(chunk_count=chunks),
            backend,
            FileExporter(),
            status=lambda _: None,
        )
        results = pipeline.run([root / "input" / "sample.wav"])
        return pipeline, results[0]

    @staticmethod
    def read(root, job_id, *parts, where="process"):
        return json.loads(root.joinpath(where, job_id, *parts).read_text(encoding="utf-8"))


class ParallelRunTests(Workspace):
    def test_two_requests_overlap_and_match_the_sequential_result(self):
        sequential_root, parallel_root = self.make_root("seq"), self.make_root("par")
        sequential = SleepingBackend()
        parallel = SleepingBackend()
        _, seq_result = self.run_job(sequential_root, sequential)
        _, par_result = self.run_job(parallel_root, parallel, parallel_requests=2)
        self.assertEqual(sequential.max_active, 1)
        self.assertEqual(parallel.max_active, 2)
        self.assertEqual(seq_result["status"], "completed")
        self.assertEqual(par_result["status"], "completed")
        seq_transcript = self.read(
            sequential_root, seq_result["job_id"], "transcript.json", where="output"
        )
        par_transcript = self.read(
            parallel_root, par_result["job_id"], "transcript.json", where="output"
        )
        self.assertEqual(seq_transcript["segments"], par_transcript["segments"])
        starts = [segment["start"] for segment in par_transcript["segments"]]
        self.assertEqual(starts, sorted(starts))
        checkpoint = self.read(parallel_root, par_result["job_id"], "checkpoint.json")
        self.assertEqual(sorted(checkpoint["chunks"], key=int), [str(i) for i in range(6)])
        self.assertEqual(checkpoint["failed"], {})
        self.assertEqual(sorted(checkpoint["attempts"], key=int), [str(i) for i in range(6)])
        report = self.read(parallel_root, par_result["job_id"], "report.json", where="output")
        self.assertEqual(report["execution"]["parallel_requests"], 2)
        intermediate = self.read(
            parallel_root, par_result["job_id"], "intermediate", "report.json", where="output"
        )
        self.assertEqual(intermediate["inference_configuration"]["parallel_requests"], 2)

    def test_concurrency_never_exceeds_the_configured_limit(self):
        for workers in (2, 3):
            backend = SleepingBackend(delay=0.03)
            root = self.make_root(f"w{workers}")
            _, result = self.run_job(root, backend, chunks=9, parallel_requests=workers)
            self.assertEqual(result["status"], "completed")
            self.assertLessEqual(backend.max_active, workers)
            self.assertGreaterEqual(backend.max_active, 2)

    def test_degenerate_chunk_walks_the_ladder_and_job_ends_incomplete(self):
        def behavior(chunk, temperature):
            if chunk.index == 1:
                raise DegenerateOutputError("loop")
            if chunk.index == 2 and temperature is None:
                raise DegenerateOutputError("loop")
            return None

        backend = SleepingBackend(delay=0.02, behavior=behavior)
        root = self.make_root()
        _, result = self.run_job(
            root,
            backend,
            chunks=5,
            parallel_requests=2,
            fallback_temperatures=(0.2,),
            split_on_failure=False,
        )
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failed_chunks"], [1])
        checkpoint = self.read(root, result["job_id"], "checkpoint.json")
        self.assertEqual(sorted(checkpoint["chunks"], key=int), ["0", "2", "3", "4"])
        self.assertEqual(list(checkpoint["failed"]), ["1"])
        self.assertEqual(
            [entry["temperature"] for entry in checkpoint["attempts"]["2"]], [0.0, 0.2]
        )
        self.assertEqual(len(checkpoint["attempts"]["1"]), 2)
        self.assertFalse((root / "output" / result["job_id"] / "transcript.json").exists())
        intermediate = self.read(
            root, result["job_id"], "intermediate", "report.json", where="output"
        )
        self.assertEqual(intermediate["status"], "incomplete")
        self.assertEqual([item["index"] for item in intermediate["failed_chunks"]], [1])

    def test_transient_failure_stops_new_work_keeps_completed_results_and_resumes(self):
        def behavior(chunk, temperature):
            if chunk.index == 3:
                raise TransientBackendError("server down")
            return None

        root = self.make_root()
        backend = SleepingBackend(delay=0.05, behavior=behavior)
        _, result = self.run_job(root, backend, chunks=8, parallel_requests=2)
        self.assertEqual(result["status"], "failed")
        self.assertIn("unavailable", result["error"])
        submitted = {call[0] for call in backend.calls}
        self.assertNotIn(5, submitted)
        self.assertNotIn(7, submitted)
        checkpoint = self.read(root, result["job_id"], "checkpoint.json")
        done = set(checkpoint["chunks"])
        self.assertNotIn("3", done)
        # Chunk 2 was in flight beside chunk 3; it completed and was saved.
        self.assertTrue({"0", "1", "2"} <= done)
        self.assertEqual(done, {str(index) for index in submitted - {3}})
        incomplete = self.read(
            root, result["job_id"], "intermediate", "report.json", where="output"
        )
        self.assertEqual(incomplete["status"], "failed")

        resumed = SleepingBackend(delay=0.01)
        _, second = self.run_job(root, resumed, chunks=8, parallel_requests=2)
        self.assertEqual(second["status"], "completed")
        self.assertEqual({call[0] for call in resumed.calls}, set(range(8)) - set(map(int, done)))
        final = self.read(root, second["job_id"], "checkpoint.json")
        self.assertEqual(len(final["chunks"]), 8)

    def test_resume_after_interrupted_parallel_run_skips_done_and_retries_failed(self):
        root = self.make_root()
        first = SleepingBackend(
            delay=0.01,
            behavior=lambda chunk, t: (
                (_ for _ in ()).throw(DegenerateOutputError("loop")) if chunk.index == 2 else None
            ),
        )
        _, result = self.run_job(
            root,
            first,
            chunks=5,
            parallel_requests=2,
            fallback_temperatures=(),
            split_on_failure=False,
        )
        self.assertEqual(result["status"], "incomplete")
        second = SleepingBackend(delay=0.01)
        _, rerun = self.run_job(
            root,
            second,
            chunks=5,
            parallel_requests=2,
            fallback_temperatures=(),
            split_on_failure=False,
        )
        self.assertEqual(rerun["status"], "completed")
        self.assertEqual([call[0] for call in second.calls], [2])


class ThreadLocalMetricsTests(unittest.TestCase):
    def test_concurrent_calls_keep_their_own_metrics(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            chunks = []
            for index in range(2):
                path = root / f"c{index}.wav"
                path.write_bytes(b"synthetic")
                chunks.append(AudioChunk(path, index * 1.0, index * 1.0 + 1.0, index))
            backend = LlamaCppBackend(response_mode="plain")
            backend._server_model_id = "m"
            barrier = threading.Barrier(2, timeout=5)

            def post(path, payload, *, purpose):
                barrier.wait()  # both requests are in flight together
                index = int(purpose.rsplit(" ", 1)[1])
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": f"text {index}"},
                        }
                    ],
                    "usage": {"completion_tokens": 100 + index, "prompt_tokens": 7},
                }

            seen: dict[int, int] = {}

            def work(chunk):
                backend.transcribe(chunk, language=None)
                time.sleep(0.02)
                seen[chunk.index] = backend.last_call_metrics()["completion_tokens"]

            with patch.object(backend, "_post_json", side_effect=post):
                threads = [threading.Thread(target=work, args=(c,)) for c in chunks]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(10)
            self.assertEqual(seen, {0: 100, 1: 101})
            self.assertIsNone(backend.last_call_metrics())  # main thread made no call

    def test_concurrent_audits_for_one_chunk_never_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "c0.wav"
            path.write_bytes(b"synthetic")
            chunk = AudioChunk(path, 0.0, 1.0, 0)
            backend = LlamaCppBackend(response_mode="plain")
            backend._server_model_id = "m"
            response = {
                "choices": [{"finish_reason": "stop", "message": {"content": "hello"}}],
                "usage": {"completion_tokens": 1},
            }
            barrier = threading.Barrier(4, timeout=5)

            def work():
                barrier.wait()
                backend._save_response_audit(
                    chunk, None, response, temperature=0.0, max_tokens=10, outcome={}
                )

            threads = [threading.Thread(target=work) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            names = sorted(item.name for item in (root / "responses").iterdir())
            self.assertEqual(
                names,
                [
                    "00000_t000.json",
                    "00000_t000_r2.json",
                    "00000_t000_r3.json",
                    "00000_t000_r4.json",
                ],
            )


class SlowExporter(FileExporter):
    def export(self, transcript, destination):
        time.sleep(0.06)
        return super().export(transcript, destination)


class RealBackend(SleepingBackend):
    name = "fake-real"  # not "mock", so a completed queued source is archived


class ReportTimingTests(Workspace):
    def test_stage_times_and_both_inference_figures_are_rendered(self):
        root = self.make_root()
        slow = 0.06
        original_archive = TranscriptionPipeline._archive_input

        def slow_archive(self, *args, **kwargs):
            time.sleep(slow)
            return original_archive(self, *args, **kwargs)

        original_refresh = pipeline_module.refresh_source

        def slow_refresh(*args, **kwargs):
            time.sleep(slow)
            return original_refresh(*args, **kwargs)

        pipeline = TranscriptionPipeline(
            make_config(root, archive_inputs=True, retries=0, parallel_requests=2),
            FakeProcessor(chunk_count=6),
            RealBackend(delay=0.05),
            SlowExporter(),
            status=lambda _: None,
        )
        with (
            patch.object(TranscriptionPipeline, "_archive_input", slow_archive),
            patch.object(pipeline_module, "refresh_source", slow_refresh),
        ):
            result = pipeline.run([root / "input" / "sample.wav"])[0]
        self.assertEqual(result["status"], "completed")
        text = (root / "output" / result["job_id"] / "run-report.md").read_text("utf-8")
        rows = {
            line.split("|")[1].strip(): line.split("|")[2].strip()
            for line in text.splitlines()
            if line.startswith("| ") and line.count("|") == 3
        }
        for label in ("Export", "Archive", "Catalog bookkeeping"):
            self.assertIn(label, rows)
            self.assertNotIn(rows[label], {"0.0 s", "n/a"}, label)
        self.assertIn("Inference stage wall time (incl. per-chunk bookkeeping)", rows)
        request_label = next(k for k in rows if k.startswith("Sum of model request times"))
        self.assertIn("overlap", request_label)
        self.assertIn("Effective concurrency:", text)
        self.assertIn("Parallel requests: 2", text)
        self.assertIn("may change greedy output", text)
        report = self.read(root, result["job_id"], "report.json", where="output")
        execution = report["execution"]
        self.assertGreater(execution["effective_concurrency"], 1.0)
        self.assertGreater(
            execution["request_seconds_sum"], execution["inference_stage_wall_seconds"]
        )
        metadata = self.read(root, result["job_id"], "metadata.json")
        stages = metadata["runs"][-1]["stages_seconds"]
        self.assertGreater(stages["catalog"], 0)
        self.assertGreater(stages["archive"], 0)

    def test_sequential_report_has_no_overlap_note(self):
        root = self.make_root()
        _, result = self.run_job(root, SleepingBackend(delay=0.01), chunks=3)
        text = (root / "output" / result["job_id"] / "run-report.md").read_text("utf-8")
        self.assertIn("Sum of model request times", text)
        self.assertNotIn("may change greedy output", text)
        self.assertNotIn("overlap", text)


class IntermediateThrottleTests(Workspace):
    def run_counting(self, root, **config):
        counts: list[str] = []
        checkpoint_writes: list[int] = []
        original = TranscriptionPipeline._write_intermediate
        original_atomic = pipeline_module.atomic_json

        def counting(self, details, segments, *, checkpoint, status="processing", error=None):
            counts.append(status)
            return original(
                self, details, segments, checkpoint=checkpoint, status=status, error=error
            )

        def atomic(path, data):
            if Path(path).name == "checkpoint.json":
                checkpoint_writes.append(len(data.get("chunks", {})))
            return original_atomic(path, data)

        pipeline = TranscriptionPipeline(
            make_config(root, archive_inputs=False, retries=0, **config),
            FakeProcessor(chunk_count=5),
            SleepingBackend(delay=0.0),
            FileExporter(),
            status=lambda _: None,
        )
        pipeline._clock = lambda: 100.0  # time never advances
        with (
            patch.object(TranscriptionPipeline, "_write_intermediate", counting),
            patch.object(pipeline_module, "atomic_json", atomic),
        ):
            result = pipeline.run([root / "input" / "sample.wav"])[0]
        return result, counts, checkpoint_writes

    def test_interval_zero_writes_after_every_chunk(self):
        result, counts, checkpoints = self.run_counting(
            self.make_root(), intermediate_interval_seconds=0.0
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(counts, ["processing"] * 5 + ["completed"])
        self.assertEqual(checkpoints, [1, 2, 3, 4, 5])

    def test_large_interval_writes_only_at_start_and_end_but_checkpoints_every_chunk(self):
        for workers in (1, 2):
            result, counts, checkpoints = self.run_counting(
                self.make_root(f"w{workers}"),
                intermediate_interval_seconds=3600.0,
                parallel_requests=workers,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(counts, ["processing", "completed"])
            self.assertEqual(sorted(checkpoints), [1, 2, 3, 4, 5])

    def test_elapsed_interval_triggers_another_write(self):
        root = self.make_root()
        counts: list[str] = []
        original = TranscriptionPipeline._write_intermediate

        def counting(self, details, segments, *, checkpoint, status="processing", error=None):
            counts.append(status)
            return original(
                self, details, segments, checkpoint=checkpoint, status=status, error=error
            )

        ticks = iter(range(0, 1000, 4))  # each clock read advances 4 s
        pipeline = TranscriptionPipeline(
            make_config(root, archive_inputs=False, intermediate_interval_seconds=10.0, retries=0),
            FakeProcessor(chunk_count=6),
            SleepingBackend(delay=0.0),
            FileExporter(),
            status=lambda _: None,
        )
        pipeline._clock = lambda: float(next(ticks))
        with patch.object(TranscriptionPipeline, "_write_intermediate", counting):
            pipeline.run([root / "input" / "sample.wav"])
        processing = counts.count("processing")
        self.assertTrue(1 < processing < 6, counts)
        self.assertEqual(counts[-1], "completed")


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = load_config(None, {})
        self.assertEqual(config.parallel_requests, 1)
        self.assertEqual(config.intermediate_interval_seconds, 10.0)

    def test_valid_values(self):
        config = load_config(
            None,
            {"backend": "llamacpp", "parallel_requests": 8, "intermediate_interval_seconds": 0},
        )
        self.assertEqual((config.parallel_requests, config.intermediate_interval_seconds), (8, 0.0))

    def test_invalid_parallel_requests(self):
        for value in (0, 9, -1, True, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_config(None, {"backend": "llamacpp", "parallel_requests": value})
        with self.assertRaisesRegex(ValueError, "llamacpp"):
            load_config(None, {"backend": "nexa", "parallel_requests": 2})

    def test_invalid_intermediate_interval(self):
        for value in (-1, float("nan"), float("inf"), True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_config(None, {"intermediate_interval_seconds": value})

    def test_toml_keys_are_accepted(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "c.toml"
            path.write_text(
                'backend = "llamacpp"\nparallel_requests = 2\nintermediate_interval_seconds = 5\n',
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual((config.parallel_requests, config.intermediate_interval_seconds), (2, 5.0))

    def test_parallelism_is_a_distinct_version_but_export_interval_is_not(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)

            def fingerprint(**overrides):
                config = make_config(root, backend="llamacpp", model="m", **overrides)
                return TranscriptionPipeline(config, FakeProcessor())._fingerprint()

            base = fingerprint()
            self.assertEqual(base, fingerprint(parallel_requests=1))
            self.assertNotEqual(base, fingerprint(parallel_requests=2))
            self.assertNotEqual(fingerprint(parallel_requests=2), fingerprint(parallel_requests=4))
            self.assertEqual(base, fingerprint(intermediate_interval_seconds=0.0))

    def test_inference_configuration_records_parallel_requests(self):
        with tempfile.TemporaryDirectory() as name:
            config = make_config(Path(name), backend="llamacpp", model="m", parallel_requests=3)
            recorded = TranscriptionPipeline(config, FakeProcessor())._inference_configuration()
        self.assertEqual(recorded["parallel_requests"], 3)


if __name__ == "__main__":
    unittest.main()
