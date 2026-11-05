"""Desktop contract checks; no GUI runtime, model or private recording required."""

import http.client
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from audio_transcript.adapters.desktop_media import WaveformCancelled, import_media, waveform
from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.adapters.ffmpeg import FFmpegProcessor, parse_progress
from audio_transcript.application.delivery import CANONICAL_FILES
from audio_transcript.application.layout import parse_version_folder, version_folder_name
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.config import DEFAULT_OUTPUT_FORMATS, load_config, parse_output_formats
from audio_transcript.desktop.bridge import DesktopBridge, selection_options
from audio_transcript.desktop.playback import PlaybackServer
from audio_transcript.desktop.results import (
    _last_packet_end,
    inspect_files,
    queued_file_outcome,
)
from audio_transcript.desktop.runtime import verify_placement
from audio_transcript.domain.models import TranscriptionError

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
    from .test_integration import RATE, tone, write_wav
except ImportError:
    from fakes import FakeBackend, FakeProcessor, make_config
    from test_integration import RATE, tone, write_wav


class DesktopContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_atomic_import_preserves_external_source_and_avoids_collisions(self):
        source = self.root / "talk.wav"
        source.write_bytes(b"unchanged original bytes")
        queue = self.root / "input"
        first = import_media(source, queue)
        second = import_media(source, queue)
        self.assertNotEqual(first, second)
        self.assertEqual(source.read_bytes(), first.read_bytes())
        self.assertEqual(source.read_bytes(), second.read_bytes())
        self.assertEqual(len(list(queue.iterdir())), 2)

    def test_incomplete_import_is_not_discovered_and_is_cleaned_after_failure(self):
        source = self.root / "talk.wav"
        source.write_bytes(b"media")
        queue = self.root / "input"

        def broken_copy(original, target, length):
            target.write(b"partial")
            pipeline = TranscriptionPipeline(make_config(self.root), FakeProcessor())
            self.assertEqual(pipeline.discover(), [])
            raise OSError("copy failed")

        with patch("audio_transcript.adapters.desktop_media.shutil.copyfileobj", broken_copy):
            with self.assertRaisesRegex(OSError, "copy failed"):
                import_media(source, queue)
        self.assertEqual(list(queue.iterdir()), [])
        self.assertEqual(source.read_bytes(), b"media")

    def test_publication_failure_keeps_original_and_existing_copy(self):
        source = self.root / "talk.wav"
        source.write_bytes(b"media")
        with patch(
            "audio_transcript.adapters.desktop_media.os.link", side_effect=OSError("no links")
        ):
            with self.assertRaises(OSError):
                import_media(source, self.root / "input")
        self.assertTrue(source.exists())
        self.assertEqual(list((self.root / "input").iterdir()), [])

    def test_full_range_has_cli_identity_and_deleting_choice_is_explicit(self):
        payload = {"language": "it", "context": "", "start": 0, "end": 100}
        options = selection_options(payload, 100, "qwen3-asr")
        self.assertIsNone(options["max_duration"])
        self.assertEqual(options["after_success"], "keep-all")
        base = make_config(self.root, language="it")
        self.assertEqual(
            TranscriptionPipeline(base, FakeProcessor())._fingerprint(),
            TranscriptionPipeline(replace(base, **options), FakeProcessor())._fingerprint(),
        )
        self.assertEqual(
            selection_options({**payload, "after_success": "delete-all"}, 100, "plain")[
                "after_success"
            ],
            "delete-all",
        )

    def test_invalid_ranges_and_partial_deletion_are_rejected(self):
        payload = {"language": "it", "start": 2, "end": 8}
        for override in (
            {"start": -1},
            {"start": float("nan")},
            {"end": True},
            {"end": 101},
            {"end": 1},
            {"after_success": "delete-all"},
            {"context": "a" * 2001},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                selection_options({**payload, **override}, 100, "plain")

    def test_start_offset_validated_and_part_of_job_identity_and_scope(self):
        base = make_config(self.root, max_duration=10)
        a = TranscriptionPipeline(base, FakeProcessor())._fingerprint()
        b = TranscriptionPipeline(replace(base, start_seconds=30), FakeProcessor())._fingerprint()
        self.assertNotEqual(a, b)
        from datetime import UTC, datetime

        folder = version_folder_name(datetime.now(UTC), "test", 10, b, 30)
        self.assertEqual(parse_version_folder(folder)["scope"], "from30s-first10s")
        for start in (-1, float("nan"), float("inf"), True, "3"):
            with self.subTest(start=start), self.assertRaises(ValueError):
                load_config(overrides={"start_seconds": start})

    def test_model_and_audio_encoder_placement_are_both_required(self):
        selection = SimpleNamespace(
            backend="cuda", device="CUDA0", run_info=lambda: {"backend": "cuda"}
        )
        log = "load_tensors: CUDA0 model buffer size = 1743 MiB\nclip_ctx: CLIP using CUDA0 backend"
        self.assertEqual(verify_placement(log, selection)["audio_encoder_device"], "CUDA0")
        for bad in (
            "CUDA0 model buffer size",
            "CLIP using CUDA0 backend",
            log.replace("CLIP using CUDA0", "CLIP using CPU"),
        ):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                verify_placement(bad, selection)

    def bridge(self):
        config_path = self.root / "config.toml"
        config_path.write_text('backend = "mock"\nvad = "none"\n', encoding="utf-8")
        bridge = DesktopBridge(config_path)
        bridge._config = make_config(self.root)
        return bridge

    def test_queue_selection_requires_recap_and_detects_source_changes(self):
        bridge = self.bridge()
        queue = self.root / "input"
        queue.mkdir()
        source = queue / "talk.wav"
        source.write_bytes(b"media")
        bridge._processor = FakeProcessor(chunk_count=10)
        files = bridge.queue()["files"]
        self.assertEqual(len(files), 1)
        bridge.select(files[0]["id"])
        payload = {"start": 0, "end": 10, "language": "it"}
        with self.assertRaisesRegex(ValueError, "Review"):
            bridge.start(payload)
        bridge.review(payload)
        with self.assertRaisesRegex(ValueError, "Review"):
            bridge.start({**payload, "language": "en"})
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            bridge.review(payload)

    def test_result_opens_exact_version_and_partial_results_are_explicit(self):
        bridge = self.bridge()
        version = self.root / "output" / "talk-id" / "version"
        partial = version / "intermediate"
        partial.mkdir(parents=True)
        bridge._consume(
            {
                "type": "result",
                "ok": False,
                "jobs": [{"job_id": "talk-id/version", "status": "incomplete"}],
            }
        )
        with patch("audio_transcript.desktop.bridge.open_folder") as opened:
            bridge.open_result()
        opened.assert_called_once_with(partial)
        (version / "transcript.txt").write_text("transcript", encoding="utf-8")
        bridge._consume({"type": "result", "ok": True, "jobs": [{"job_id": "talk-id/version"}]})
        self.assertEqual(bridge._result_folder, version)
        self.assertTrue(bridge.can_close())
        bridge._running = True
        self.assertFalse(bridge.can_close())

    def test_worker_crash_is_reported_and_window_is_released(self):
        bridge = self.bridge()
        bridge._running = True
        process = SimpleNamespace(
            stdin=io.StringIO(), stdout=io.StringIO(""), wait=lambda: 9, poll=lambda: 9
        )
        with patch("audio_transcript.desktop.bridge.subprocess.Popen", return_value=process):
            bridge._work({})
        self.assertFalse(bridge.status()["running"])
        self.assertFalse(bridge.status()["result"]["ok"])
        self.assertIn("without a result", bridge.status()["result"]["error"]["message"])

    def test_stray_output_lines_do_not_stop_a_run_or_hide_its_result(self):
        bridge = self.bridge()
        bridge._running = True
        lines = '3\n"text"\n[1, 2]\n{"type": "warning", "data": "not-an-object"}\n'
        lines += '{"type": "result", "ok": true, "jobs": ["x", 4]}\n'
        process = SimpleNamespace(
            stdin=io.StringIO(),
            stdout=io.StringIO(lines),
            wait=lambda: 0,
            poll=lambda: 0,
        )
        with patch("audio_transcript.desktop.bridge.subprocess.Popen", return_value=process):
            bridge._work({})
        status = bridge.status()
        self.assertFalse(status["running"])
        self.assertTrue(status["result"]["ok"])

    def launch(self, *argv):
        from audio_transcript.desktop.__main__ import main

        errors = io.StringIO()
        with (
            patch.object(sys, "argv", ["sbobinai-desktop", *argv]),
            patch("sys.stderr", errors),
        ):
            return main(), errors.getvalue()

    def test_launcher_explains_a_missing_configuration_without_opening_a_window(self):
        # A stand-in module keeps this test independent of the optional pywebview package.
        with patch.dict(sys.modules, {"webview": SimpleNamespace()}):
            code, errors = self.launch("--config", str(self.root / "absent.toml"))
        self.assertEqual(code, 1)
        self.assertIn("Could not open the desktop app", errors)
        self.assertIn("README installation", errors)

    def test_launcher_explains_a_missing_optional_ui_dependency(self):
        with patch.dict(sys.modules, {"webview": None}):
            code, errors = self.launch("--config", str(self.root / "absent.toml"))
        self.assertEqual(code, 1)
        self.assertIn("pip install -e", errors)
        self.assertIn("[desktop]", errors)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
class DesktopMediaTests(unittest.TestCase):
    """Actual media preparation with generated tones; no real inference."""

    def test_selected_audio_samples_and_waveform_match_the_requested_interval(self):
        from audio_transcript.domain.models import SegmentationSettings

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tone.wav"
            samples = [0] * RATE + tone(1) + [0] * RATE
            write_wav(source, samples)
            before = source.read_bytes()
            processor = FFmpegProcessor()
            prepared = processor.prepare(
                source,
                root / "chunks",
                sample_rate=RATE,
                segmentation=SegmentationSettings(detector="none"),
                start_seconds=1,
                max_duration=1,
            )
            self.assertAlmostEqual(prepared.duration, 1)
            self.assertEqual(prepared.chunks[0].start, 0)
            import wave

            with wave.open(str(prepared.chunks[0].path), "rb") as wav:
                self.assertEqual(wav.readframes(RATE), before[-2 * RATE * 2 : -RATE * 2])
            preview = waveform(processor, source, root / "preview", bins=30)
            self.assertEqual(len(preview["peaks"]), 30)
            self.assertGreater(max(preview["peaks"][11:19]), 0.1)
            self.assertLess(max(preview["peaks"][:9]), 0.02)
            self.assertEqual(source.read_bytes(), before)

    def test_bounded_pipeline_keeps_input_even_with_delete_all_and_exports_offset(self):
        from audio_transcript.adapters.exporters import FileExporter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "tone.wav"
            source.parent.mkdir()
            write_wav(source, tone(3))
            backend = FakeBackend()
            backend.name = "llamacpp"  # synthetic contract check, never a real-inference claim
            cfg = make_config(root, start_seconds=1, max_duration=1, after_success="delete-all")
            pipeline = TranscriptionPipeline(
                cfg, FFmpegProcessor(), backend, FileExporter(), status=lambda _: None
            )
            result = pipeline.run([source])[0]
            self.assertEqual(result["status"], "completed")
            self.assertTrue(source.exists())
            payload = json.loads(
                (cfg.output_dir / result["job_id"] / "transcript.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["audio_metadata"]["selection_start_seconds"], 1)
            self.assertEqual(payload["segments"][0]["start"], 0)
            self.assertTrue(any("relative" in warning for warning in payload["warnings"]))

    def test_isolated_mock_worker_returns_structured_result(self):
        """Mock end-to-end: real FFmpeg, explicitly synthetic model output."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tone.wav"
            write_wav(source, tone(1))
            config = root / "config.toml"
            config.write_text('backend="mock"\nvad="none"\nmax_duration=0.1\n', encoding="utf-8")
            overrides = {
                key: str(root / key)
                for key in ("input_dir", "process_dir", "output_dir", "processed_dir")
            }
            overrides.update({"max_duration": None, "start_seconds": 0, "monitor": False})
            process = subprocess.run(
                [sys.executable, "-m", "audio_transcript.desktop.worker"],
                input=json.dumps(
                    {"config": str(config), "source": str(source), "overrides": overrides}
                ),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads(process.stdout.splitlines()[-1])
            self.assertTrue(result["ok"])
            transcript = json.loads(
                Path(result["jobs"][0]["transcript"])
                .with_suffix(".json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(transcript["duration"], 1)
            self.assertEqual(transcript["backend"], "mock")
            self.assertTrue(source.exists())


class DesktopReviewTests(unittest.TestCase):
    """Retention default, background waveform and review playback; headless, no GUI runtime."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        config_path = self.root / "config.toml"
        config_path.write_text('backend = "mock"\nvad = "none"\n', encoding="utf-8")
        self.bridge = DesktopBridge(config_path)
        self.bridge._config = make_config(self.root)
        self.bridge._processor = FakeProcessor(chunk_count=10)
        self.addCleanup(self.bridge.shutdown)
        self.queue = self.root / "input"
        self.queue.mkdir()

    def select(self, path):
        entries = self.bridge.queue()["files"]
        file_id = next(e["id"] for e in entries if e["name"] == path.name)
        return self.bridge.select(file_id)

    def pick(self, picked):
        """Run import_file with a mocked native picker returning ``picked``."""
        self.bridge._window = SimpleNamespace(create_file_dialog=lambda *a, **k: [str(picked)])
        fake_webview = SimpleNamespace(FileDialog=SimpleNamespace(OPEN=1))
        with patch.dict(sys.modules, {"webview": fake_webview}):
            result = self.bridge.import_file()
        return next(f for f in result["files"] if f["id"] == result["imported_id"])

    # ---- Change 1: retention default depends on how the file reached the queue
    def test_found_file_defaults_to_keep_and_imported_copy_to_delete(self):
        found = self.queue / "found.wav"
        found.write_bytes(b"already here")
        self.assertEqual(self.select(found)["default_after_success"], "keep-all")
        external = self.root / "external" / "picked.wav"
        external.parent.mkdir()
        external.write_bytes(b"outside the queue")
        entry = self.pick(external)
        selected = self.bridge.select(entry["id"])
        self.assertEqual(selected["default_after_success"], "delete-all")
        self.assertEqual(external.read_bytes(), b"outside the queue")
        self.assertEqual(self.select(found)["default_after_success"], "keep-all")

    def test_picking_a_file_already_in_the_queue_is_not_recorded_as_an_import(self):
        inside = self.queue / "inside.wav"
        inside.write_bytes(b"queued")
        entry = self.pick(inside)
        self.assertEqual(entry["name"], "inside.wav")
        self.assertEqual(self.bridge._imported, {})
        self.assertEqual(self.bridge.select(entry["id"])["default_after_success"], "keep-all")

    # ---- Change 2: background waveform
    def test_waveform_cancel_kills_ffmpeg_and_frees_the_source(self):
        source = self.root / "held.wav"
        source.write_bytes(b"media")
        ready = self.root / "ready"
        code = (
            "import sys,time; f=open(sys.argv[1],'rb'); open(sys.argv[2],'w').close(); "
            "time.sleep(120)"
        )
        children = []
        real_popen = subprocess.Popen

        def holder(command, **kwargs):
            child = real_popen([sys.executable, "-c", code, str(source), str(ready)], **kwargs)
            children.append(child)
            return child

        processor = SimpleNamespace(ffmpeg="ffmpeg", probe=lambda path: SimpleNamespace(duration=1))
        cancel, outcome = threading.Event(), []

        def run():
            try:
                waveform(processor, source, self.root / "scratch", cancel=cancel)
            except WaveformCancelled:
                outcome.append("cancelled")

        with patch("audio_transcript.adapters.desktop_media.subprocess.Popen", holder):
            worker = threading.Thread(target=run)
            worker.start()
            deadline = time.monotonic() + 20
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(ready.exists())
            cancel.set()
            worker.join(15)
        self.assertEqual(outcome, ["cancelled"])
        self.assertIsNotNone(children[0].poll())  # the process exited before cleanup finished
        source.rename(self.root / "renamed.wav")  # fails on Windows while a handle is open
        self.assertEqual(list((self.root / "scratch").iterdir()), [])

    def background_preview(self, started, release):
        """A stand-in for waveform() that blocks until cancelled, like a long decode."""

        def fake(processor, source, scratch, bins=800, cancel=None):
            self.assertFalse(self.bridge._operation.locked(), "decode must not hold the lock")
            started.set()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if cancel.is_set():
                    raise WaveformCancelled
                time.sleep(0.02)
            raise AssertionError("the decode was never cancelled")

        return fake

    def test_preview_does_not_hold_the_operation_lock_and_select_cancels_it(self):
        source = self.queue / "talk.wav"
        source.write_bytes(b"media")
        info = self.select(source)
        started, never = threading.Event(), threading.Event()
        results = []
        with patch(
            "audio_transcript.desktop.bridge.waveform", self.background_preview(started, never)
        ):
            thread = threading.Thread(target=lambda: results.append(self.bridge.preview()))
            thread.start()
            self.assertTrue(started.wait(10))
            # Navigation stays possible during the decode.
            payload = {"start": 0, "end": 10, "language": "it"}
            self.assertEqual(self.bridge.review(payload)["file"], "talk.wav")
            self.bridge.select(info["id"])  # cancels and waits for the decode to stop
            thread.join(10)
        self.assertEqual(results, [{"cancelled": True}])

    def test_start_cancels_a_running_waveform_before_the_worker_may_archive(self):
        source = self.queue / "talk.wav"
        source.write_bytes(b"media")
        self.select(source)
        payload = {"start": 0, "end": 10, "language": "it"}
        self.bridge.review(payload)
        started, never, results = threading.Event(), threading.Event(), []
        with patch(
            "audio_transcript.desktop.bridge.waveform", self.background_preview(started, never)
        ):
            thread = threading.Thread(target=lambda: results.append(self.bridge.preview()))
            thread.start()
            self.assertTrue(started.wait(10))

            def finish(request):
                with self.bridge._state_lock:
                    self.bridge._running = False

            with patch.object(self.bridge, "_work", finish):
                self.bridge.start(payload)
            self.assertFalse(thread.is_alive(), "start() returned while FFmpeg could still run")
            thread.join(10)
        self.assertEqual(results, [{"cancelled": True}])

    def test_can_close_stops_a_running_waveform(self):
        source = self.queue / "talk.wav"
        source.write_bytes(b"media")
        self.select(source)
        started, never, results = threading.Event(), threading.Event(), []
        with patch(
            "audio_transcript.desktop.bridge.waveform", self.background_preview(started, never)
        ):
            thread = threading.Thread(target=lambda: results.append(self.bridge.preview()))
            thread.start()
            self.assertTrue(started.wait(10))
            self.assertTrue(self.bridge.can_close())
            thread.join(10)
        self.assertEqual(results, [{"cancelled": True}])

    def test_waveform_cache_is_bounded_and_keyed_by_source_identity(self):
        calls = []

        def fake(processor, source, scratch, bins=800, cancel=None):
            calls.append(source.name)
            return {"duration": 10, "peaks": [0.1]}

        with patch("audio_transcript.desktop.bridge.waveform", fake):
            for index in range(10):
                path = self.queue / f"f{index}.wav"
                path.write_bytes(b"media")
                self.select(path)
                self.bridge.preview()
            self.assertEqual(len(calls), 10)
            self.assertEqual(len(self.bridge._peaks), 8)
            self.bridge.preview()  # the last file is cached
            self.assertEqual(len(calls), 10)
            self.select(self.queue / "f0.wav")  # evicted
            self.bridge.preview()
            self.assertEqual(len(calls), 11)

    # ---- optional MP4 preview: refused before Start, not after the transcription
    def test_mp4_is_refused_for_audio_only_recordings_and_missing_encoders(self):
        source = self.queue / "talk.wav"
        source.write_bytes(b"media")
        info = self.select(source)
        self.assertFalse(info["has_video"])
        payload = {"start": 0, "end": 10, "language": "it", "output_formats": ["txt", "mp4"]}
        with self.assertRaisesRegex(ValueError, "video stream"):
            self.bridge.review(payload)
        self.bridge._processor.video = True
        self.assertTrue(self.select(source)["has_video"])
        self.assertEqual(self.bridge.review(payload)["options"]["output_formats"], ("txt", "mp4"))

        def missing():
            raise TranscriptionError("missing encoder: libx264")

        self.bridge._processor.check_video_encoder = missing
        self.bridge._encoder_problem = None
        self.assertFalse(self.select(source)["mp4_encoder"])
        with self.assertRaisesRegex(ValueError, "libx264"):
            self.bridge.review(payload)

    # ---- Change 3: review playback
    def finished_job(self, *, start=0.0, length=10.0, status="completed"):
        """A completed mock job beside a selected queued recording."""
        source = self.queue / "talk.wav"
        source.write_bytes(b"queued media bytes")
        self.select(source)
        job = "talk-id/version"
        output = self.root / "output" / job
        output.mkdir(parents=True)
        cues = [
            {"start": 0.0, "end": 2.0, "text": "first <b>cue</b>"},
            {"start": 2.0, "end": 4.0, "text": "second cue"},
        ]
        meta = {"selection_start_seconds": start, "duration_seconds": length}
        (output / "transcript.json").write_text(
            json.dumps({"segments": cues, "audio_metadata": meta}), encoding="utf-8"
        )
        self.bridge._consume(
            {"type": "result", "ok": True, "jobs": [{"job_id": job, "status": status}]}
        )
        return source, output

    def test_playback_candidates_follow_the_defined_order_and_reject_other_files(self):
        source, output = self.finished_job()
        archived = self.root / "processed" / "talk-id" / "talk.wav"
        archived.parent.mkdir(parents=True)
        shutil.copy2(source, archived)
        original = self.root / "external" / "original.wav"
        original.parent.mkdir()
        shutil.copy2(source, original)
        stat = original.stat()
        self.bridge._imported[source.resolve()] = (original, stat.st_size, stat.st_mtime_ns)
        for name in ("video.mp4", "audio.mp3", "audio.flac"):
            (output / name).write_bytes(b"derived")
        self.assertTrue(self.bridge.status()["can_review"])
        data = self.bridge.playback()
        self.assertEqual(
            [c["label"] for c in data["candidates"]],
            [
                "the queued recording",
                "the archived recording",
                "the original you imported",
                "the 720p preview video",
                "the exported MP3 audio",
                "the exported FLAC audio",
            ],
        )
        self.assertEqual(data["cues"][0]["text"], "first <b>cue</b>")
        # Each review gets fresh URLs; a reused URL once made the WebView keep stale audio-only media.
        self.assertNotEqual(
            data["candidates"][0]["url"], self.bridge.playback()["candidates"][0]["url"]
        )
        self.assertEqual({c["offset"] for c in data["candidates"]}, {0.0})
        # A same-named file with other content must never be offered under these subtitles.
        archived.write_bytes(b"a different recording")
        original.write_bytes(b"x")
        labels = [c["label"] for c in self.bridge.playback()["candidates"]]
        self.assertNotIn("the archived recording", labels)
        self.assertNotIn("the original you imported", labels)
        source.unlink()  # the queued copy was deleted; derived files remain the fallback
        labels = [c["label"] for c in self.bridge.playback()["candidates"]]
        self.assertEqual(
            labels,
            ["the 720p preview video", "the exported MP3 audio", "the exported FLAC audio"],
        )

    def test_bounded_selection_plays_from_the_offset_and_stops_at_its_end(self):
        source, output = self.finished_job(start=3.0, length=4.0)
        (output / "audio.mp3").write_bytes(b"derived")
        candidates = self.bridge.playback()["candidates"]
        self.assertEqual((candidates[0]["offset"], candidates[0]["end"]), (3.0, 7.0))
        self.assertEqual((candidates[1]["offset"], candidates[1]["end"]), (0.0, None))
        self.assertEqual(self.bridge.playback()["selection_start"], 3.0)

    def test_review_is_offered_only_for_completed_or_skipped_jobs(self):
        self.finished_job(status="incomplete")
        self.assertFalse(self.bridge.status()["can_review"])
        with self.assertRaises(ValueError):
            self.bridge.playback()
        self.bridge._consume(
            {
                "type": "result",
                "ok": True,
                "jobs": [{"job_id": "talk-id/version", "status": "skipped"}],
            }
        )
        self.assertTrue(self.bridge.status()["can_review"])

    def test_playback_stop_releases_the_served_file(self):
        source, _ = self.finished_job()
        data = self.bridge.playback()
        self.bridge.playback_select(0)
        url = data["candidates"][0]["url"]
        host, _, path = url.removeprefix("http://").partition("/")
        connection = http.client.HTTPConnection(host)
        connection.request("GET", "/" + path)
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.read(4)  # leave the response open, like a paused player
        self.bridge.playback_stop()
        connection.close()
        source.rename(self.root / "moved.wav")  # no handle remains
        again = http.client.HTTPConnection(host)
        again.request("GET", "/" + path)
        self.assertEqual(again.getresponse().status, 404)
        again.close()


class PlaybackServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.file = Path(self.temp.name) / "clip.mp4"
        self.file.write_bytes(bytes(range(256)) * 4)  # 1024 bytes
        self.server = PlaybackServer()
        self.addCleanup(self.server.close)
        self.server.serve(self.file)
        self.host, _, path = self.server.base_url.removeprefix("http://").partition("/")
        self.path = "/" + path

    def fetch(self, method="GET", path=None, headers=None):
        connection = http.client.HTTPConnection(self.host)
        connection.request(method, path or self.path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response, body

    def test_full_response_and_head(self):
        response, body = self.fetch()
        self.assertEqual((response.status, len(body)), (200, 1024))
        self.assertEqual(response.getheader("Accept-Ranges"), "bytes")
        self.assertEqual(response.getheader("Content-Type"), "video/mp4")
        response, body = self.fetch("HEAD")
        self.assertEqual((response.status, body), (200, b""))
        self.assertEqual(response.getheader("Content-Length"), "1024")

    def test_ranges(self):
        data = self.file.read_bytes()
        for header, expected, span in (
            ("bytes=10-19", data[10:20], "10-19/1024"),
            ("bytes=1000-", data[1000:], "1000-1023/1024"),
            ("bytes=-24", data[-24:], "1000-1023/1024"),
            ("bytes=1020-5000", data[1020:], "1020-1023/1024"),
        ):
            with self.subTest(header=header):
                response, body = self.fetch(headers={"Range": header})
                self.assertEqual(response.status, 206)
                self.assertEqual(body, expected)
                self.assertEqual(response.getheader("Content-Range"), "bytes " + span)
        for header in ("bytes=1024-", "bytes=2000-2100", "bytes=-0"):
            with self.subTest(header=header):
                response, _ = self.fetch(headers={"Range": header})
                self.assertEqual(response.status, 416)
                self.assertEqual(response.getheader("Content-Range"), "bytes */1024")

    def test_only_the_tokenised_route_is_served(self):
        token_path = self.path
        for path in (
            "/",
            "/wrong-token/media",
            token_path + "/extra",
            token_path.replace("/media", "/../media"),
            "/../" + str(self.file),
            f"/{self.file.name}",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.fetch(path=path)[0].status, 404)
        self.assertEqual(self.fetch(path=token_path + "?c=2")[0].status, 200)
        self.assertEqual(self.fetch("POST")[0].status, 501)

    def test_nothing_is_served_after_release(self):
        self.server.release()
        self.assertEqual(self.fetch()[0].status, 404)
        self.file.unlink()  # no handler keeps the file open


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
class PreviewVideoTests(unittest.TestCase):
    """Real FFmpeg on generated lavfi video; mock inference only."""

    @classmethod
    def setUpClass(cls):
        cls.processor = FFmpegProcessor()
        try:
            cls.processor.check_video_encoder()
        except Exception as exc:
            raise unittest.SkipTest(f"FFmpeg lacks the preview encoders: {exc}") from exc

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self, name="clip.mp4", size="1920x1080", seconds=4):
        path = self.root / name
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(path),
            ],
            check=True,
        )  # fmt: skip
        return path

    def streams(self, path):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,codec_type,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return json.loads(out.stdout)["streams"]

    def test_1080p_is_downscaled_to_720p_for_the_selected_interval_with_soft_subtitles(self):
        source = self.video()
        before = source.read_bytes()
        srt = self.root / "transcript.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,500\nHello\n", encoding="utf-8")
        info = self.processor.export_preview_video(
            source, self.root / "video.mp4", start_seconds=1, max_duration=2, subtitles=srt
        )
        self.assertEqual(info.video_height, 720)
        self.assertEqual(info.codec, "aac")
        self.assertAlmostEqual(info.duration, 2, delta=0.3)
        kinds = {s["codec_type"]: s["codec_name"] for s in self.streams(self.root / "video.mp4")}
        self.assertEqual(kinds, {"video": "h264", "audio": "aac", "subtitle": "mov_text"})
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse((self.root / "video.mp4.part").exists())

    def test_a_480p_source_is_not_upscaled_and_subtitles_are_optional(self):
        info = self.processor.export_preview_video(
            self.video("small.mp4", "854x480"), self.root / "video.mp4"
        )
        self.assertEqual(info.video_height, 480)
        self.assertAlmostEqual(info.duration, 4, delta=0.4)
        self.assertNotIn(
            "subtitle", {s["codec_type"] for s in self.streams(self.root / "video.mp4")}
        )

    def test_audio_only_sources_are_rejected_before_any_render(self):
        tone_path = self.root / "tone.wav"
        write_wav(tone_path, tone(2))
        self.assertFalse(self.processor.probe(tone_path).has_video)
        with self.assertRaisesRegex(TranscriptionError, "no video"):
            self.processor.export_preview_video(tone_path, self.root / "video.mp4")
        self.assertFalse((self.root / "video.mp4").exists())

    def test_cover_art_is_not_a_video_stream(self):
        art = self.root / "art.mp3"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=1",
                "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-map", "0", "-map", "1",
                "-c:a", "libmp3lame", "-c:v", "mjpeg", "-frames:v", "1",
                "-disposition:v", "attached_pic", str(art),
            ],
            check=True,
        )  # fmt: skip
        self.assertFalse(self.processor.probe(art).has_video)

    def test_mp4_is_off_by_default_and_never_changes_inference_identity(self):
        self.assertNotIn("mp4", DEFAULT_OUTPUT_FORMATS)
        self.assertNotIn("video.mp4", CANONICAL_FILES)
        self.assertNotIn("mp4", make_config(self.root).output_formats)
        self.assertEqual(parse_output_formats("mp4,txt"), ("txt", "mp4"))
        base = make_config(self.root)
        with_video = replace(base, output_formats=(*base.output_formats, "mp4"))
        self.assertEqual(
            TranscriptionPipeline(base, FakeProcessor())._fingerprint(),
            TranscriptionPipeline(with_video, FakeProcessor())._fingerprint(),
        )

    def run_pipeline(self, source, **overrides):
        backend = FakeBackend()
        backend.name = "llamacpp"  # generated fixtures exercise retention, not inference
        cfg = make_config(
            self.root, output_formats=("srt", "mp4"), archive_intermediate=True, **overrides
        )
        pipeline = TranscriptionPipeline(
            cfg, self.processor, backend, FileExporter(), status=lambda _: None
        )
        return pipeline.run([source])[0], cfg, backend

    def test_pipeline_renders_the_preview_and_an_audio_only_source_fails_before_inference(self):
        (self.root / "input").mkdir()
        source = self.video("input/clip.mp4", seconds=3)
        result, cfg, backend = self.run_pipeline(source)
        self.assertEqual(result["status"], "completed", result)
        rendered = cfg.output_dir / result["job_id"] / "video.mp4"
        self.assertEqual(self.processor.probe(rendered).video_height, 720)
        self.assertTrue(any(cfg.processed_dir.rglob("clip.mp4")))  # archived after success
        audio = self.root / "input" / "tone.wav"
        write_wav(audio, tone(2))
        result, cfg, backend = self.run_pipeline(audio)
        self.assertEqual(result["status"], "failed")
        self.assertIn("audio only", result["error"])
        self.assertEqual(backend.calls, [])  # refused before any transcription
        self.assertTrue(audio.exists())

    def test_a_failed_render_retains_the_source_and_a_rerun_skips_inference(self):
        (self.root / "input").mkdir()
        source = self.video("input/clip.mp4", seconds=3)
        failing = FFmpegProcessor()

        def broken(*args, **kwargs):
            raise TranscriptionError("render failed")

        failing.export_preview_video = broken
        backend = FakeBackend()
        backend.name = "llamacpp"
        cfg = make_config(self.root, output_formats=("srt", "mp4"), after_success="delete-all")
        result = TranscriptionPipeline(
            cfg, failing, backend, FileExporter(), status=lambda _: None
        ).run([source])[0]
        self.assertEqual(result["status"], "failed")
        self.assertTrue(source.exists())
        self.assertIn("render failed", result["error"])
        calls = len(backend.calls)
        result = TranscriptionPipeline(
            cfg, self.processor, backend, FileExporter(), status=lambda _: None
        ).run([source])[0]
        self.assertIn(result["status"], {"completed", "skipped"}, result)
        self.assertEqual(len(backend.calls), calls)  # outputs restored without new inference
        self.assertTrue(any(cfg.output_dir.rglob("video.mp4")))


class Recorder:
    """A minimal event bus that keeps everything the pipeline emits."""

    run_id = None
    sink_errors = 0

    def __init__(self):
        self.events = []

    def emit(self, type_, data=None, *, job_id=None):
        self.events.append((type_, dict(data or {})))

    def of(self, type_, **match):
        return [
            d for t, d in self.events if t == type_ and all(d.get(k) == v for k, v in match.items())
        ]


class ProgressParsingTests(unittest.TestCase):
    def test_progress_blocks_give_percent_and_ignore_noise(self):
        text = "frame=10\nout_time_us=1500000\nspeed=3x\nprogress=continue\nout_time_us=N/A\n"
        info = parse_progress(text, 6.0)
        self.assertEqual((info["done_s"], info["percent"]), (1.5, 25.0))
        self.assertEqual(parse_progress("out_time_ms=2000000\n", 4.0)["percent"], 50.0)
        self.assertEqual(parse_progress("out_time_us=9000000\n", 4.0)["percent"], 100.0)
        self.assertIsNone(parse_progress("frame=1\nprogress=continue\n", 4.0))

    def test_a_missing_duration_never_invents_a_percentage(self):
        for total in (None, 0, float("nan")):
            info = parse_progress("out_time_us=2000000\n", total)
            self.assertEqual(info["done_s"], 2.0)
            self.assertIsNone(info["percent"])
            self.assertIsNone(info["eta_s"])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
class ReuseAndProgressTests(unittest.TestCase):
    """Re-running identical settings with MP4 newly added, with real FFmpeg and mock inference."""

    @classmethod
    def setUpClass(cls):
        cls.processor = FFmpegProcessor()
        try:
            cls.processor.check_video_encoder()
        except Exception as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        self.source = self.root / "input" / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=10:duration=3",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(self.source),
            ],
            check=True,
        )  # fmt: skip
        self.backend = FakeBackend()

    def run_job(self, formats, processor=None, **overrides):
        events = Recorder()
        config = make_config(self.root, output_formats=formats, **overrides)
        pipeline = TranscriptionPipeline(
            config, processor or self.processor, self.backend, FileExporter(), status=lambda _: None
        )
        pipeline.attach(events=events)
        return pipeline.run([self.source])[0], events, config

    def test_reusing_a_transcription_for_a_new_mp4_reports_every_stage(self):
        first, _, _ = self.run_job(("srt",))
        self.assertEqual(first["status"], "completed", first)
        calls = len(self.backend.calls)
        result, events, config = self.run_job(("srt", "mp4"))
        self.assertEqual(result["status"], "skipped", result)
        self.assertEqual(len(self.backend.calls), calls)  # no model work
        reused = events.of("job.reused")
        self.assertEqual(len(reused), 1)
        self.assertIn("probe", reused[0]["stages"])  # the earlier run's timings travel along
        self.assertIsNotNone(reused[0]["completed_at"])
        for stage in ("video_export", "packaging"):
            self.assertEqual(len(events.of("stage.started", stage=stage)), 1, stage)
            finished = events.of("stage.finished", stage=stage)
            self.assertEqual(len(finished), 1, stage)
            self.assertGreaterEqual(finished[0]["seconds"], 0)
        progress = events.of("stage.progress", stage="video_export")
        self.assertTrue(progress)
        self.assertEqual(progress[-1]["percent"], 100.0)
        done = events.of("job.finished")[0]
        self.assertTrue(done["reused"])
        self.assertIn("video_export", done["timings"]["stages"])
        self.assertIsNotNone(done["counts"])
        self.assertTrue((config.output_dir / result["job_id"] / "video.mp4").is_file())
        self.assertTrue(self.source.exists())

    def test_the_encoder_path_is_recorded_and_a_fall_through_is_a_visible_warning(self):
        self.run_job(("srt",))
        processor = FFmpegProcessor()
        good = {
            "encoder": "libx264", "gpu_decode": False, "pre": [],
            "codec": ["-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-pix_fmt", "yuv420p"],
        }  # fmt: skip
        broken = {"encoder": "h264_nvenc", "gpu_decode": False, "pre": [],
                  "codec": ["-c:v", "no_such_encoder"]}  # fmt: skip
        processor._video_attempts = lambda: [broken, good]
        result, events, config = self.run_job(("srt", "mp4"), processor)
        self.assertEqual(result["status"], "skipped", result)
        warning = events.of("warning", code="video_encoder_fallback")
        self.assertEqual(len(warning), 1)
        self.assertIn("using the CPU encoder", warning[0]["message"])
        finished = events.of("stage.finished", stage="video_export")[0]
        self.assertEqual((finished["encoder"], finished["gpu_decode"]), ("libx264", False))
        progress = events.of("stage.progress", stage="video_export")
        self.assertEqual(progress[-1]["encoder"], "libx264")
        record = json.loads(
            (config.process_dir / result["job_id"] / "delivery.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["video_render"], {"encoder": "libx264", "gpu_decode": False})
        # The results row names the encoder that really ran.
        rows = inspect_files(
            config.output_dir / result["job_id"], config, result["job_id"], processor
        )["files"]
        note = next(r["note"] for r in rows if r["name"] == "video.mp4")
        self.assertTrue(note.endswith("CPU (x264)"), note)
        # A later run that reuses the file keeps the record.
        again, _, _ = self.run_job(("srt", "mp4"), processor)
        self.assertEqual(again["status"], "skipped")
        record = json.loads(
            (config.process_dir / result["job_id"] / "delivery.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["video_render"]["encoder"], "libx264")

    def test_a_failed_render_in_the_reuse_path_fails_before_any_deletion(self):
        self.run_job(("srt",))
        broken = FFmpegProcessor()

        def fail(*args, **kwargs):
            raise TranscriptionError("render failed")

        broken.export_preview_video = fail
        result, events, config = self.run_job(("srt", "mp4"), broken, after_success="delete-all")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(self.source.exists())
        self.assertFalse(any(config.processed_dir.rglob("clip.mp4")))
        self.assertFalse(events.of("stage.finished", stage="video_export"))

    def test_the_render_reports_throttled_progress_and_a_final_hundred_percent(self):
        reports = []
        info = self.processor.export_preview_video(
            self.source, self.root / "video.mp4", progress=reports.append
        )
        self.assertEqual(info.video_height, 720)
        self.assertTrue(reports)
        self.assertEqual(reports[-1]["percent"], 100.0)
        self.assertTrue(all(0 <= r["percent"] <= 100 for r in reports))


class VideoEncoderChainTests(unittest.TestCase):
    """GPU-first MP4 rendering: selection, fall-through, no upscale. GPU cases skip without NVENC."""

    LISTING = " V....D libx264 x\n A....D aac x\n S..... mov_text x\n V....D h264_nvenc x\n"

    def processor(self, listing=None, nvenc=None):
        processor = FFmpegProcessor()
        if listing is not None:
            processor._encoder_listing = lambda: listing
        if nvenc is not None:
            processor.nvenc_usable = lambda: nvenc
        return processor

    def test_check_video_encoder_accepts_libx264_or_a_working_nvenc_only(self):
        only_x264 = " V....D libx264 x\n A....D aac x\n S..... mov_text x\n"
        only_nvenc = " V....D h264_nvenc x\n A....D aac x\n S..... mov_text x\n"
        self.processor(only_x264, nvenc=False).check_video_encoder()
        self.processor(only_nvenc, nvenc=True).check_video_encoder()
        for listing, nvenc in (
            (only_nvenc, False),
            ("", False),
            (only_x264.replace("aac", ""), False),
        ):
            with self.assertRaisesRegex(TranscriptionError, "cannot render"):
                self.processor(listing, nvenc=nvenc).check_video_encoder()

    def test_libx264_alone_never_opens_a_gpu_session(self):
        processor = FFmpegProcessor()
        processor._encoder_listing = lambda: self.LISTING

        def forbidden():
            raise AssertionError("the GPU probe must not run when libx264 is available")

        processor.nvenc_usable = forbidden
        processor.check_video_encoder()

    def test_the_chain_is_gpu_decode_then_cpu_decode_nvenc_then_libx264(self):
        chain = self.processor(nvenc=True)._video_attempts()
        self.assertEqual(
            [(a["encoder"], a["gpu_decode"]) for a in chain],
            [("h264_nvenc", True), ("h264_nvenc", False), ("libx264", False)],
        )
        self.assertIn("-hwaccel", chain[0]["pre"])
        for attempt in chain:  # never upscale: the height is capped, not forced
            self.assertIn("min(720,ih)", " ".join(attempt["codec"]))
            self.assertNotIn("=-2:720", " ".join(attempt["codec"]))
        self.assertEqual(
            [a["encoder"] for a in self.processor(nvenc=False)._video_attempts()], ["libx264"]
        )

    def generate(self, directory, size="1280x720", seconds=3):
        path = Path(directory) / f"src-{size}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=10:duration={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(path),
            ],
            check=True,
        )  # fmt: skip
        return path

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_a_failing_first_encoder_falls_through_visibly_and_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = FFmpegProcessor()
            try:
                processor.check_video_encoder()
            except TranscriptionError as exc:
                self.skipTest(str(exc))
            good = {
                "encoder": "libx264", "gpu_decode": False, "pre": [],
                "codec": ["-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-pix_fmt", "yuv420p"],
            }  # fmt: skip
            broken = {"encoder": "h264_nvenc", "gpu_decode": True, "pre": [],
                      "codec": ["-c:v", "no_such_encoder"]}  # fmt: skip
            processor._video_attempts = lambda: [broken, good]
            warnings, reports = [], []
            destination = Path(directory) / "video.mp4"
            info = processor.export_preview_video(
                self.generate(directory), destination, progress=reports.append, warn=warnings.append
            )
            self.assertEqual(info.render, {"encoder": "libx264", "gpu_decode": False})
            self.assertEqual(len(warnings), 1)
            self.assertIn("NVENC with GPU decoding failed", warnings[0])
            self.assertIn("CPU encoder", warnings[0])
            self.assertEqual(
                sorted(p.name for p in Path(directory).glob("video.mp4*")), ["video.mp4"]
            )
            zeros = [r for r in reports if r["percent"] == 0.0]
            self.assertEqual(len(zeros), 2)  # the retry restarts the bar
            self.assertEqual(zeros[0]["encoder"], "h264_nvenc")
            self.assertEqual(zeros[1]["encoder"], "libx264")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_when_every_encoder_fails_nothing_is_published(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = FFmpegProcessor()
            broken = {"encoder": "libx264", "gpu_decode": False, "pre": [],
                      "codec": ["-c:v", "no_such_encoder"]}  # fmt: skip
            processor._video_attempts = lambda: [broken]
            with self.assertRaises(TranscriptionError):
                processor.export_preview_video(
                    self.generate(directory), Path(directory) / "video.mp4"
                )
            self.assertEqual(list(Path(directory).glob("video.mp4*")), [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_each_available_path_downscales_without_upscaling(self):
        processor = FFmpegProcessor()
        attempts = processor._video_attempts()
        if not any(a["encoder"] == "h264_nvenc" for a in attempts):
            self.skipTest("NVENC is not usable here; only the CPU path can be exercised")
        for attempt in attempts:
            for size, expected in (("1920x1080", 720), ("854x480", 480)):
                with self.subTest(path=(attempt["encoder"], attempt["gpu_decode"]), size=size):
                    with tempfile.TemporaryDirectory() as directory:
                        processor._video_attempts = lambda a=attempt: [a]
                        info = processor.export_preview_video(
                            self.generate(directory, size, 3), Path(directory) / "video.mp4"
                        )
                        self.assertEqual(info.video_height, expected)
                        self.assertEqual(info.codec, "aac")
                        self.assertAlmostEqual(info.duration, 3, delta=0.5)
                        self.assertEqual(info.render["encoder"], attempt["encoder"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_the_nvenc_interval_and_soft_subtitles_match_the_selection(self):
        processor = FFmpegProcessor()
        gpu = [a for a in processor._video_attempts() if a["encoder"] == "h264_nvenc"]
        if not gpu:
            self.skipTest("NVENC is not usable here")
        with tempfile.TemporaryDirectory() as directory:
            srt = Path(directory) / "transcript.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
            processor._video_attempts = lambda: gpu[:1]
            info = processor.export_preview_video(
                self.generate(directory, "1920x1080", 6), Path(directory) / "video.mp4",
                start_seconds=1, max_duration=2, subtitles=srt,
            )  # fmt: skip
            self.assertAlmostEqual(info.duration, 2, delta=0.4)
            self.assertEqual(info.video_height, 720)
            self.assertTrue(info.render["gpu_decode"])
            kinds = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                 str(Path(directory) / "video.mp4")],
                capture_output=True, text=True, check=True,
            ).stdout.split()  # fmt: skip
            self.assertIn("subtitle", kinds)

    def test_nvenc_probe_is_real_and_cached_per_process(self):
        processor = FFmpegProcessor()
        calls = []
        processor._encoder_listing = lambda: " V....D h264_nvenc x"
        with (
            patch.dict("audio_transcript.adapters.ffmpeg._NVENC_PROBE", clear=True),
            patch("audio_transcript.adapters.ffmpeg.subprocess.run") as run,
        ):
            run.side_effect = lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=1)
            self.assertFalse(processor.nvenc_usable())
            self.assertFalse(processor.nvenc_usable())
        self.assertEqual(len(calls), 1)
        self.assertIn("h264_nvenc", calls[0][0])
        self.assertIn("lavfi", calls[0][0])
        processor.ffmpeg = "other-ffmpeg"  # a different executable is probed on its own
        processor._encoder_listing = lambda: ""
        self.assertFalse(processor.nvenc_usable())


class ResultInspectionTests(unittest.TestCase):
    """What the results screen may claim about each file."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = make_config(self.root)
        self.job = "talk-id/version"
        self.folder = self.config.output_dir / self.job
        self.folder.mkdir(parents=True)
        self.canonical = self.config.process_dir / self.job / "artifacts"
        self.canonical.mkdir(parents=True)

    def write(self, name, data, canonical=True):
        (self.folder / name).write_bytes(data)
        if canonical:
            (self.canonical / name).write_bytes(data)

    def states(self, processor=None):
        result = inspect_files(
            self.folder, self.config, self.job, processor or FakeProcessor(), partial=False
        )
        return {row["name"]: row["state"] for row in result["files"]}, result

    def test_text_reports_are_complete_but_never_called_verified(self):
        self.write("transcript.txt", b"hello")
        self.write("transcript.srt", b"1\n")
        self.write("transcript.json", b'{"audio_metadata": {}}')
        states, result = self.states()
        self.assertEqual(set(states.values()), {"complete"})
        self.assertEqual(result["problems"], 0)

    def test_empty_corrupt_and_diverging_files_are_reported_not_raised(self):
        self.write("transcript.txt", b"")
        self.write("transcript.json", b"{broken", canonical=False)
        self.write("transcript.md", b"public copy")
        (self.canonical / "transcript.md").write_bytes(b"different canonical size")
        self.write("intermediate.zip", b"PK-not-a-zip", canonical=False)
        states, result = self.states()
        self.assertEqual(
            states,
            {
                "transcript.txt": "problem",
                "transcript.json": "problem",
                "transcript.md": "problem",
                "intermediate.zip": "problem",
            },
        )
        self.assertEqual(result["problems"], 4)

    def test_zip_is_verified_by_reading_it_back_and_partial_jobs_are_labelled(self):
        with zipfile.ZipFile(self.folder / "intermediate.zip", "w") as archive:
            archive.writestr("a.txt", "a")
        states, _ = self.states()
        self.assertEqual(states["intermediate.zip"], "verified")
        partial = inspect_files(self.folder, self.config, self.job, FakeProcessor(), partial=True)
        self.assertEqual({r["state"] for r in partial["files"]}, {"partial"})

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_audio_is_reprobed_and_a_truncated_file_is_flagged(self):
        processor = FFmpegProcessor()
        source = self.root / "tone.wav"
        write_wav(source, tone(4))
        (self.canonical / "transcript.json").write_text(
            json.dumps({"audio_metadata": {"duration_seconds": 4.0}}), encoding="utf-8"
        )
        processor.export_selection_audio(
            source, self.folder / "audio.mp3", sample_rate=16000, audio_format="mp3"
        )
        states, _ = self.states(processor)
        self.assertEqual(states["audio.mp3"], "verified")
        data = (self.folder / "audio.mp3").read_bytes()
        (self.folder / "audio.mp3").write_bytes(data[: len(data) // 3])
        states, result = self.states(processor)
        self.assertEqual(states["audio.mp3"], "problem")
        self.assertEqual(result["problems"], 1)
        (self.folder / "audio.mp3").write_bytes(b"not audio at all")
        self.assertEqual(self.states(processor)[0]["audio.mp3"], "problem")

    def test_last_packet_time_tolerates_the_trailing_comma_of_packets_with_side_data(self):
        processor = SimpleNamespace(ffprobe="ffprobe")
        for output, expected in (
            ("1.0,0.5\n59.953923,0.023220\n59.977143,0.022857,\r\n", 60.0),
            ("3.0,N/A\n", 3.0),
            ("", None),
        ):
            with patch(
                "audio_transcript.desktop.results.subprocess.run",
                return_value=SimpleNamespace(stdout=output),
            ):
                end = _last_packet_end(processor, Path("x.mp4"))
            if expected is None:
                self.assertIsNone(end)
            else:
                self.assertAlmostEqual(end, expected, places=3)

    def test_queued_file_outcome_comes_from_the_job_records(self):
        queued = self.root / "input" / "talk.wav"
        queued.parent.mkdir(parents=True)

        def meta(**fields):
            (self.config.process_dir / self.job / "metadata.json").write_text(
                json.dumps(fields), encoding="utf-8"
            )
            return queued_file_outcome(self.config, self.job, queued, "completed")

        self.assertEqual(
            meta(source_disposition={"policy": "delete-all", "original_deleted": True})["state"],
            "deleted",
        )
        archived = self.root / "processed" / "talk-id" / "talk.wav"
        archived.parent.mkdir(parents=True)
        archived.write_bytes(b"x")
        self.assertEqual(meta(archived_source_path=str(archived))["state"], "archived")
        archived.unlink()
        self.assertEqual(meta(archived_source_path=str(archived))["state"], "gone")
        self.assertEqual(meta()["state"], "gone")
        queued.write_bytes(b"x")
        bounded = meta(scope={"kind": "bounded", "label": "first10s"}, backend="llamacpp")
        self.assertEqual(bounded["state"], "kept")
        self.assertIn("only part", bounded["text"])
        self.assertIn("mock", meta(scope={"kind": "full"}, backend="mock")["text"])
        self.assertIn("disk full", meta(archive_warning="disk full")["text"])
        meta(scope={"kind": "full"})
        failed = queued_file_outcome(self.config, self.job, queued, "failed")
        self.assertIn("did not finish", failed["text"])


class DesktopFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        config_path = self.root / "config.toml"
        config_path.write_text('backend = "mock"\nvad = "none"\n', encoding="utf-8")
        self.bridge = DesktopBridge(config_path)
        self.bridge._config = make_config(self.root)
        self.bridge._processor = FakeProcessor(chunk_count=10)

    def feed(self, kind, **data):
        self.bridge._consume({"type": kind, "data": data})

    def test_stages_are_collected_and_progress_never_carries_over_between_phases(self):
        self.feed("stage.started", stage="video_export")
        self.feed("stage.progress", stage="video_export", percent=43.0, eta_s=130, speed_x=14)
        self.assertEqual(self.bridge.status()["stage_progress"]["percent"], 43.0)
        self.feed("stage.finished", stage="video_export", seconds=12.5)
        self.feed("stage.started", stage="archive")
        status = self.bridge.status()
        self.assertNotIn("stage_progress", status)
        self.feed("stage.finished", stage="archive", seconds=0.2)
        self.feed("stage.finished", stage="speech_detection", seconds=None, reused=True)
        self.assertEqual(self.bridge.status()["stages"], {"video_export": 12.5, "archive": 0.2})

    def test_reuse_is_visible_in_the_phase_and_the_state(self):
        self.feed(
            "job.reused", version="v1", completed_at="2026-11-05T10:00:00Z", stages={"probe": 1}
        )
        status = self.bridge.status()
        self.assertEqual(status["phase"], "reuse")
        self.assertEqual(status["reused"]["version"], "v1")

    def test_result_details_reads_the_actual_version_folder(self):
        queue = self.root / "input"
        queue.mkdir()
        source = queue / "talk.wav"
        source.write_bytes(b"media")
        file_id = self.bridge.queue()["files"][0]["id"]
        self.bridge.select(file_id)
        folder = self.root / "output" / "talk-id" / "version"
        folder.mkdir(parents=True)
        (folder / "transcript.txt").write_text("hi", encoding="utf-8")
        self.bridge._consume(
            {
                "type": "result",
                "ok": True,
                "jobs": [{"job_id": "talk-id/version", "status": "completed"}],
            }
        )
        details = self.bridge.result_details()
        self.assertEqual(details["files"][0]["state"], "complete")
        self.assertEqual(details["queued"]["state"], "kept")
        self.bridge._running = True
        with self.assertRaises(ValueError):
            self.bridge.result_details()


class QueuedFileOutcomeRecordsTests(unittest.TestCase):
    """The outcome must follow the files on disk, not stale or misread metadata."""

    def outcome(self, root, meta, *, queued_exists=False, status="completed"):
        from types import SimpleNamespace

        from audio_transcript.desktop.results import queued_file_outcome

        job = root / "process" / "source" / "version"
        job.mkdir(parents=True, exist_ok=True)
        (job / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        queue = root / "input"
        queue.mkdir(exist_ok=True)
        queued = queue / "talk.wav"
        if queued_exists:
            queued.write_bytes(b"x")
        config = SimpleNamespace(process_dir=root / "process", input_dir=queue)
        return queued_file_outcome(config, "source/version", queued, status)

    def test_full_mock_job_is_not_described_as_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.outcome(
                Path(directory),
                {"scope": {"kind": "full", "label": "full"}, "backend": "mock"},
                queued_exists=True,
            )
        self.assertEqual(result["state"], "kept")
        self.assertIn("mock", result["text"])
        self.assertNotIn("part of the recording", result["text"])

    def test_bounded_job_is_described_as_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.outcome(
                Path(directory),
                {"scope": {"kind": "bounded", "label": "first20s"}, "backend": "llamacpp"},
                queued_exists=True,
            )
        self.assertIn("part of the recording", result["text"])

    def test_stale_deletion_record_does_not_hide_a_file_still_in_input(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.outcome(
                Path(directory),
                {
                    "scope": {"kind": "full"},
                    "source_disposition": {"policy": "delete-all", "original_deleted": True},
                },
                queued_exists=True,
            )
        self.assertEqual(result["state"], "kept")

    def test_stale_deletion_record_does_not_hide_a_new_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archived = root / "processed.wav"
            archived.write_bytes(b"x")
            result = self.outcome(
                root,
                {
                    "archived_source_path": str(archived),
                    "source_disposition": {"policy": "delete-all", "original_deleted": True},
                },
            )
        self.assertEqual(result["state"], "archived")

    def test_deleted_copy_is_reported_as_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.outcome(
                Path(directory),
                {"source_disposition": {"policy": "delete-all", "original_deleted": True}},
            )
        self.assertEqual(result["state"], "deleted")


if __name__ == "__main__":
    unittest.main()
