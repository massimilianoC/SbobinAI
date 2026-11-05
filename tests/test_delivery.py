"""Output selection, ZIP recovery and source-retention contracts; no real inference."""

import json
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.adapters.ffmpeg import FFmpegProcessor
from audio_transcript.application.catalog import version_summary
from audio_transcript.application.delivery import archive_intermediate
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.config import DEFAULT_OUTPUT_FORMATS, load_config, parse_output_formats
from audio_transcript.domain.models import (
    DegenerateOutputError,
    MediaInfo,
    Segment,
    TranscriptionError,
)

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
    from .test_integration import tone, write_wav
except ImportError:
    from fakes import FakeBackend, FakeProcessor, make_config
    from test_integration import tone, write_wav


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input" / "generated.wav"
        self.source.parent.mkdir()
        self.source.write_bytes(b"synthetic fixture; no real media or inference")

    def run_job(self, *, config=None, processor=None, backend=None):
        config = config or make_config(
            self.root, output_formats=("txt",), archive_intermediate=True
        )
        processor = processor or FakeProcessor()
        backend = backend or FakeBackend()
        pipeline = TranscriptionPipeline(
            config, processor, backend, FileExporter(), status=lambda _: None
        )
        return pipeline.run([self.source])[0], config, backend

    def test_selection_validates_and_does_not_change_inference_identity(self):
        self.assertEqual(parse_output_formats("srt,txt,srt"), ("txt", "srt"))
        for bad in ([], "", ["pdf"], [True], 1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_output_formats(bad)
        config = load_config(overrides={"output_formats": "txt,srt", "archive_intermediate": True})
        self.assertEqual(config.output_formats, ("txt", "srt"))
        base = make_config(self.root)
        self.assertEqual(
            TranscriptionPipeline(base, FakeProcessor())._fingerprint(),
            TranscriptionPipeline(
                replace(base, output_formats=("srt",), archive_intermediate=True), FakeProcessor()
            )._fingerprint(),
        )

    def test_txt_only_hides_other_exports_but_keeps_canonical_data_and_catalog(self):
        result, cfg, _ = self.run_job()
        self.assertEqual(result["status"], "completed")
        output = cfg.output_dir / result["job_id"]
        self.assertEqual(
            {path.name for path in output.iterdir()}, {"transcript.txt", "intermediate.zip"}
        )
        with zipfile.ZipFile(output / "intermediate.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("report.json", archive.namelist())
            self.assertIn("run-report.md", archive.namelist())
        canonical = cfg.process_dir / result["job_id"] / "artifacts"
        self.assertTrue((canonical / "transcript.json").exists())
        self.assertTrue((cfg.process_dir / result["job_id"] / "checkpoint.json").exists())
        summary = version_summary(cfg, *result["job_id"].split("/"))
        self.assertEqual(summary["completion_percent"], 100)
        self.assertEqual(summary["chunks"]["ok"], 2)
        self.assertIsNone(summary["artifacts"]["report"])
        self.assertTrue(summary["artifacts"]["transcript"].endswith("transcript.txt"))
        self.assertTrue(self.source.exists())  # mock output never disposes of input

    def test_changing_formats_reuses_completed_inference_and_rematerializes_outputs(self):
        first, cfg, _ = self.run_job()
        for formats in (("srt", "report-json"), DEFAULT_OUTPUT_FORMATS):
            backend = FakeBackend()
            result, selected, _ = self.run_job(
                config=replace(cfg, output_formats=formats), backend=backend
            )
            self.assertEqual(result["job_id"], first["job_id"])
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(backend.calls, [])
            self.assertTrue((selected.output_dir / result["job_id"] / "transcript.srt").exists())
        self.assertTrue((cfg.output_dir / first["job_id"] / "transcript.txt").exists())

    def test_failed_chunks_keep_readable_intermediates_and_input(self):
        cfg = make_config(
            self.root,
            output_formats=("txt",),
            archive_intermediate=True,
            after_success="delete-all",
        )

        def answer(chunk, temperature):
            if chunk.index == 0:
                raise DegenerateOutputError("unusable synthetic response")
            return [Segment(chunk.start, chunk.end, "synthetic fixture")]

        cfg = replace(cfg, split_on_failure=False, fallback_temperatures=())
        result, _, _ = self.run_job(config=cfg, backend=FakeBackend(behavior=answer))
        self.assertEqual(result["status"], "incomplete")
        output = cfg.output_dir / result["job_id"]
        self.assertTrue((output / "intermediate" / "transcript.txt").exists())
        self.assertFalse((output / "intermediate.zip").exists())
        self.assertTrue(self.source.exists())

    def test_zip_verification_failure_keeps_loose_files(self):
        output = self.root / "output" / "version"
        intermediate = output / "intermediate"
        intermediate.mkdir(parents=True)
        (intermediate / "transcript.txt").write_text("recover me", encoding="utf-8")
        with patch(
            "audio_transcript.application.delivery.zipfile.ZipFile.testzip", return_value="bad"
        ):
            with self.assertRaisesRegex(OSError, "verification failed"):
                archive_intermediate(output)
        self.assertEqual((intermediate / "transcript.txt").read_text(), "recover me")
        self.assertFalse((output / "intermediate.zip").exists())
        archive_intermediate(output)
        self.assertFalse(intermediate.exists())
        self.assertTrue((output / "intermediate.zip").exists())

    def test_packaging_failure_is_visible_and_does_not_destroy_recoverable_exports(self):
        with patch(
            "audio_transcript.application.delivery.archive_intermediate",
            side_effect=OSError("disk full"),
        ):
            result, cfg, _ = self.run_job()
        self.assertEqual(result["status"], "completed")
        self.assertIn("disk full", result["delivery_warning"])
        self.assertTrue((cfg.output_dir / result["job_id"] / "intermediate").is_dir())
        meta = json.loads(
            (cfg.process_dir / result["job_id"] / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertIn("disk full", meta["delivery_warning"])

    def test_audio_export_failure_blocks_source_deletion(self):
        processor = FakeProcessor()

        def fail(*args, **kwargs):
            destination = args[1]
            meta = json.loads(
                (destination.parent.parent / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["status"], "processing")
            raise TranscriptionError("encoder unavailable")

        processor.export_selection_audio = fail
        backend = FakeBackend()
        backend.name = "llamacpp"  # synthetic contract fixture, not real inference
        cfg = make_config(
            self.root,
            output_formats=("txt", "mp3"),
            archive_intermediate=True,
            after_success="delete-all",
        )
        result, _, _ = self.run_job(config=cfg, processor=processor, backend=backend)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(self.source.exists())
        self.assertIn("encoder unavailable", result["error"])
        self.assertFalse(any(cfg.processed_dir.rglob("generated.wav")))

    def test_unexpected_audio_codec_blocks_source_deletion(self):
        processor = FakeProcessor()

        def wrong_audio(source, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"invalid fake delivery")
            return MediaInfo(2, 16000, 1, "wrong-codec")

        processor.export_selection_audio = wrong_audio
        backend = FakeBackend()
        backend.name = "llamacpp"
        cfg = make_config(self.root, output_formats=("txt", "mp3"), after_success="delete-all")
        result, _, _ = self.run_job(config=cfg, processor=processor, backend=backend)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(self.source.exists())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
class AudioDeliveryTests(unittest.TestCase):
    """Actual FFmpeg extraction with generated media and explicitly fake inference."""

    def test_selected_mp3_and_flac_are_probed_and_bound_to_the_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            write_wav(source, tone(4))
            original = source.read_bytes()
            processor = FFmpegProcessor()
            for audio_format in ("mp3", "flac"):
                info = processor.export_selection_audio(
                    source,
                    root / f"audio.{audio_format}",
                    sample_rate=16000,
                    audio_format=audio_format,
                    start_seconds=1,
                    max_duration=2,
                )
                self.assertEqual(info.codec, audio_format)
                self.assertAlmostEqual(info.duration, 2, delta=0.15)
            self.assertEqual(source.read_bytes(), original)

    def test_verified_audio_outputs_exist_before_a_full_queued_copy_is_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "generated.wav"
            source.parent.mkdir()
            write_wav(source, tone(2))
            backend = FakeBackend()
            backend.name = "llamacpp"  # only exercises retention with generated fixtures
            cfg = make_config(
                root,
                output_formats=("srt", "mp3", "flac"),
                archive_intermediate=True,
                after_success="delete-all",
            )
            result = TranscriptionPipeline(
                cfg, FFmpegProcessor(), backend, FileExporter(), status=lambda _: None
            ).run([source])[0]
            self.assertEqual(result["status"], "completed", result)
            self.assertFalse(source.exists())
            output = cfg.output_dir / result["job_id"]
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"transcript.srt", "audio.mp3", "audio.flac", "intermediate.zip"},
            )
            self.assertEqual(FFmpegProcessor().probe(output / "audio.flac").codec, "flac")
            self.assertEqual(FFmpegProcessor().probe(output / "audio.mp3").codec, "mp3")

    def test_external_source_and_bounded_source_are_never_deleted_with_selected_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external.wav"
            queued = root / "input" / "generated.wav"
            queued.parent.mkdir()
            for source in (external, queued):
                write_wav(source, tone(3))
                backend = FakeBackend()
                backend.name = "llamacpp"
                cfg = make_config(
                    root,
                    output_formats=("txt", "mp3"),
                    archive_intermediate=True,
                    after_success="delete-all",
                    max_duration=1 if source == queued else None,
                )
                result = TranscriptionPipeline(
                    cfg, FFmpegProcessor(), backend, FileExporter(), status=lambda _: None
                ).run([source])[0]
                self.assertEqual(result["status"], "completed", result)
                self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
