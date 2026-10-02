"""Event bus, file sinks and pipeline instrumentation (synthetic, no network or GPU)."""

import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application.events import (
    CallbackSink,
    EventBus,
    NullBus,
    new_run_id,
)
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.application.progress import ProgressTracker, RunState
from audio_transcript.application.session import RunSession
from audio_transcript.domain.models import DegenerateOutputError, Segment

try:
    from .fakes import FakeProcessor, make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeProcessor, make_config

SENTINEL = "SENTINEL-TRANSCRIPT-TEXT-9f3a"


class SentinelBackend:
    """Thread-safe fake whose transcript text must never reach an event or a log."""

    name = "mock"

    def __init__(self, delay=0.0, degenerate_first=False):
        self.delay = delay
        self.degenerate_first = degenerate_first
        self._seen: set[int] = set()
        self._lock = threading.Lock()

    def check(self):
        return None

    def close(self):
        return None

    def transcribe(self, chunk, *, language, temperature=None, use_prompt=True):
        if self.delay:
            time.sleep(self.delay)
        if self.degenerate_first:
            with self._lock:
                first = chunk.index not in self._seen
                self._seen.add(chunk.index)
            if first:
                raise DegenerateOutputError(f"repetition loop {SENTINEL}")
        return [Segment(chunk.start, chunk.end, f"{SENTINEL} {chunk.index}")]


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def observed_run(root, *, chunks=3, backend=None, config=None, sources=None, lines=None):
    config = config or make_config(root)
    source = root / "input" / "talk.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_bytes(b"synthetic media bytes")
    session = RunSession(config, command="run", ui_mode="plain", queue_size=1, status_out=None)
    session.start()
    status = (lambda message: lines.append(message)) if lines is not None else (lambda m: None)
    pipeline = TranscriptionPipeline(
        config,
        FakeProcessor(chunks),
        backend or SentinelBackend(),
        FileExporter(),
        status=status,
        events=session.bus,
    )
    results = pipeline.run(sources or [source])
    session.finish(0, results)
    session.close()
    return results, session


class BusTests(unittest.TestCase):
    def test_run_id_format(self):
        self.assertRegex(new_run_id(), r"^\d{8}T\d{6}Z-[0-9a-f]{4}$")

    def test_seq_is_monotonic_across_threads(self):
        seen = []
        bus = EventBus("r1", [CallbackSink(seen.append)])

        def work():
            for _ in range(200):
                bus.emit("log", {"level": "info", "message": "x"})

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([event.seq for event in seen], list(range(1, 1601)))
        stamps = [event.elapsed_s for event in seen]
        self.assertEqual(stamps, sorted(stamps))

    def test_failing_sink_is_counted_and_reported_once(self):
        calls = []

        class Broken:
            def handle(self, event):
                raise RuntimeError("disk full")

            def close(self):
                return None

        stderr = io.StringIO()
        bus = EventBus("r1", [Broken(), CallbackSink(calls.append)], stderr=stderr)
        for _ in range(5):
            bus.emit("log", {"message": "x"})
        self.assertEqual(len(calls), 5)
        self.assertEqual(bus.sink_errors, 5)
        self.assertEqual(stderr.getvalue().count("event sink Broken failed"), 1)

    def test_non_finite_numbers_are_serialised_as_null(self):
        seen = []
        bus = EventBus("r1", [CallbackSink(lambda e: seen.append(e.to_json()))])
        bus.emit("progress", {"eta_s": float("inf"), "nested": {"x": float("nan")}})
        data = json.loads(seen[0])["data"]
        self.assertIsNone(data["eta_s"])
        self.assertIsNone(data["nested"]["x"])

    def test_null_bus_is_inert(self):
        bus = NullBus()
        bus.emit("log", {"message": "x"})
        self.assertIsNone(bus.run_id)


class ProgressTests(unittest.TestCase):
    def test_tracker_counts_and_eta(self):
        now = [0.0]
        tracker = ProgressTracker({0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}, clock=lambda: now[0])
        tracker.seed({"0": [{"x": 1}]}, {})
        now[0] = 5.0
        history = [
            {
                "stage": "temperature",
                "outcome": "ok",
                "metrics": {"completion_tokens": 40, "mean_token_prob": 0.9, "logprob_tokens": 40},
            }
        ]
        self.assertEqual(tracker.finish_chunk(1, [object()], history), "ok")
        snap = tracker.snapshot()
        self.assertEqual(snap["done_chunks"], 2)
        self.assertEqual(snap["percent"], 50.0)
        self.assertEqual(snap["fallbacks"], 1)
        self.assertEqual(snap["tokens"], 40)
        self.assertEqual(snap["throughput_x"], 2.0)  # 10 s of audio in 5 s of wall time
        self.assertEqual(snap["eta_s"], 10.0)
        self.assertEqual(snap["confidence_mean"], 0.9)
        self.assertEqual(tracker.finish_chunk(2, [], []), "no_speech")
        self.assertEqual(tracker.finish_chunk(3, None, []), "failed")
        self.assertEqual(tracker.counters()["failed"], 1)

    def test_run_state_reduces_events(self):
        seen = []
        bus = EventBus("r1", [CallbackSink(seen.append)])
        bus.emit("run.started", {"command": "run", "model": "m", "queue_size": 2})
        bus.emit("job.started", {"source": "a.wav", "index": 1, "total": 2}, job_id="s/v")
        bus.emit("stage.started", {"stage": "probe"}, job_id="s/v")
        bus.emit("stage.finished", {"stage": "probe", "seconds": 0.5}, job_id="s/v")
        bus.emit("log", {"level": "error", "message": "boom"}, job_id="s/v")
        state = RunState()
        for event in seen:
            state.apply(event)
        snap = state.snapshot()
        self.assertEqual(snap["current_job"]["stages"]["probe"]["state"], "done")
        self.assertEqual(snap["warnings"], 1)
        self.assertEqual(snap["queue_size"], 2)


class PipelineEventTests(unittest.TestCase):
    def test_event_stream_files_and_ordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, session = observed_run(root, chunks=4)
            self.assertEqual(results[0]["status"], "completed")
            events = read_events(session.paths.events)
            seqs = [event["seq"] for event in events]
            self.assertEqual(seqs, list(range(1, len(events) + 1)))
            types = [event["type"] for event in events]
            self.assertEqual(types[0], "run.started")
            self.assertEqual(types[-1], "run.finished")
            for needed in (
                "job.started",
                "stage.started",
                "stage.finished",
                "job.prepared",
                "chunk.started",
                "chunk.attempt",
                "chunk.finished",
                "progress",
                "log",
                "job.finished",
            ):
                self.assertIn(needed, types)
            self.assertLess(types.index("job.started"), types.index("job.prepared"))
            self.assertLess(types.index("job.prepared"), types.index("chunk.started"))
            self.assertLess(types.index("job.finished"), types.index("run.finished"))
            self.assertEqual({event["run_id"] for event in events}, {session.run_id})
            last = [e for e in events if e["type"] == "progress"][-1]["data"]
            self.assertEqual(last["done_chunks"], 4)
            self.assertEqual(last["percent"], 100.0)
            finished = [e for e in events if e["type"] == "job.finished"][0]
            self.assertEqual(finished["data"]["status"], "completed")
            self.assertEqual(finished["data"]["counts"]["ok"], 4)
            self.assertTrue(
                all(a.startswith("output/") for a in finished["data"]["artifacts"]),
                finished["data"]["artifacts"],
            )
            status = json.loads(session.paths.status.read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "finished")
            self.assertEqual(status["current_job"]["progress"]["done_chunks"], 4)
            latest = json.loads(session.paths.latest.read_text(encoding="utf-8"))
            self.assertEqual(latest["run_id"], session.run_id)
            self.assertEqual(latest["state"], "finished")
            self.assertEqual(latest["paths"]["events"], f"runs/{session.run_id}/events.jsonl")
            self.assertTrue(session.paths.log.is_file())

    def test_parallel_chunks_keep_ordered_unique_seq(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, backend="llamacpp", parallel_requests=3, model="m")
            results, session = observed_run(
                root, chunks=9, backend=SentinelBackend(delay=0.02), config=config
            )
            self.assertEqual(results[0]["status"], "completed")
            events = read_events(session.paths.events)
            seqs = [event["seq"] for event in events]
            self.assertEqual(seqs, sorted(set(seqs)))
            elapsed = [event["elapsed_s"] for event in events]
            self.assertEqual(elapsed, sorted(elapsed))
            started = [e["data"] for e in events if e["type"] == "chunk.started"]
            self.assertEqual(len(started), 9)
            self.assertGreater(max(item["in_flight_count"] for item in started), 1)
            self.assertLessEqual(max(item["in_flight_count"] for item in started), 3)
            percents = [e["data"]["percent"] for e in events if e["type"] == "progress"]
            self.assertEqual(percents, sorted(percents))

    def test_per_job_events_append_across_a_resume_with_distinct_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = observed_run(root, chunks=2)
            config = make_config(root)
            results, second = observed_run(root, chunks=2, config=config)
            self.assertEqual(results[0]["status"], "skipped")
            self.assertNotEqual(first.run_id, second.run_id)
            job_id = read_events(first.paths.events)[1]["job_id"]
            folder, version = job_id.split("/")
            copy = read_events(root / "process" / folder / version / "events.jsonl")
            self.assertEqual({e["run_id"] for e in copy}, {first.run_id, second.run_id})
            self.assertTrue(all(e["job_id"] == job_id for e in copy))
            self.assertNotIn("resource.sample", {e["type"] for e in copy})
            metadata = json.loads(
                (root / "process" / folder / version / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["runs"][0]["run_id"], first.run_id)

    def test_plain_log_matches_status_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines: list[str] = []
            _, session = observed_run(root, chunks=3, lines=lines)
            logged = [
                line.split("] ", 1)[1]
                for line in session.paths.log.read_text(encoding="utf-8").splitlines()
                if " [info] " in line or " [warning] " in line or " [error] " in line
            ]
            self.assertEqual(logged, lines)
            text = session.paths.log.read_text(encoding="utf-8")
            self.assertIn("[summary]", text)
            self.assertIn("[run] finished", text)

    def test_no_transcript_text_or_private_paths_in_any_event_or_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, session = observed_run(
                root, chunks=3, backend=SentinelBackend(degenerate_first=True)
            )
            job_dir = next((root / "process").glob("talk-*"))
            blobs = [
                session.paths.events.read_text(encoding="utf-8"),
                session.paths.log.read_text(encoding="utf-8"),
                session.paths.status.read_text(encoding="utf-8"),
                *(p.read_text(encoding="utf-8") for p in job_dir.rglob("events.jsonl")),
            ]
            for blob in blobs:
                self.assertNotIn(SENTINEL, blob)
                self.assertNotIn(str(root), blob)
                self.assertNotIn(str(root).replace("\\", "/"), blob)
            events = read_events(session.paths.events)
            attempts = [e["data"] for e in events if e["type"] == "chunk.attempt"]
            self.assertTrue(any(a["outcome"] == "DegenerateOutputError" for a in attempts))
            self.assertTrue(any(a["stage"] == "temperature" for a in attempts))

    def test_a_failing_sink_never_breaks_a_job(self):
        class Broken:
            def handle(self, event):
                raise OSError("sink exploded")

            def close(self):
                raise OSError("close exploded")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = root / "input" / "talk.wav"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"synthetic media bytes")
            stderr = io.StringIO()
            bus = EventBus("r1", [Broken()], stderr=stderr)
            pipeline = TranscriptionPipeline(
                config,
                FakeProcessor(2),
                SentinelBackend(),
                FileExporter(),
                status=lambda m: None,
                events=bus,
            )
            results = pipeline.run([source])
            bus.close()
            self.assertEqual(results[0]["status"], "completed")
            self.assertGreater(bus.sink_errors, 10)
            self.assertEqual(stderr.getvalue().count("sink exploded"), 1)

    def test_default_pipeline_has_no_events_and_unchanged_run_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = root / "input" / "talk.wav"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"synthetic media bytes")
            pipeline = TranscriptionPipeline(
                config, FakeProcessor(2), SentinelBackend(), FileExporter(), status=lambda m: None
            )
            results = pipeline.run([source])
            self.assertEqual(results[0]["status"], "completed")
            self.assertFalse((root / "process" / "runs").exists())
            metadata = next((root / "process").glob("talk-*/*/metadata.json"))
            run = json.loads(metadata.read_text(encoding="utf-8"))["runs"][0]
            self.assertNotIn("run_id", run)

    def test_failed_job_emits_finished_with_short_error(self):
        class Exploding(SentinelBackend):
            def transcribe(self, chunk, *, language, temperature=None, use_prompt=True):
                raise ValueError("unexpected")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, session = observed_run(root, chunks=2, backend=Exploding())
            self.assertEqual(results[0]["status"], "failed")
            events = read_events(session.paths.events)
            finished = [e for e in events if e["type"] == "job.finished"]
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["data"]["status"], "failed")
            self.assertIn("error", finished[0]["data"])
            self.assertEqual(events[-1]["type"], "run.finished")

    def test_file_size_rejection_is_a_started_finished_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, max_file_size=1)
            _, session = observed_run(root, chunks=2, config=config)
            types = [e["type"] for e in read_events(session.paths.events)]
            self.assertEqual(types.count("job.started"), 1)
            self.assertEqual(types.count("job.finished"), 1)


if __name__ == "__main__":
    unittest.main()
