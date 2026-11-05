import json
import struct
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters.ffmpeg import FFmpegProcessor
from audio_transcript.adapters.mock import MockBackend
from audio_transcript.domain.models import (
    AudioChunk,
    SegmentationSettings,
    TranscriptionError,
    VoiceActivity,
)


class FFmpegProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        self.source.touch()
        self.processor = FFmpegProcessor("ffmpeg-test", "ffprobe-test")

    def tearDown(self):
        self.temp.cleanup()

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_probe_extracts_audio_metadata(self, run):
        run.return_value = types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"duration": "12.5"},
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "sample_rate": "48000",
                            "channels": 2,
                        }
                    ],
                }
            ),
            stderr="",
        )
        info = self.processor.probe(self.source)
        self.assertEqual(
            (info.duration, info.sample_rate, info.channels, info.codec), (12.5, 48000, 2, "aac")
        )
        self.assertIsInstance(run.call_args.args[0], list)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_probe_rejects_media_without_audio(self, run):
        run.return_value = types.SimpleNamespace(
            returncode=0, stdout='{"format":{"duration":"2"},"streams":[]}', stderr=""
        )
        with self.assertRaisesRegex(TranscriptionError, "no audio"):
            self.processor.probe(self.source)

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_probe_rejects_malformed_metadata_actionably(self, run):
        run.return_value = types.SimpleNamespace(returncode=0, stdout='{"streams":null}', stderr="")
        with self.assertRaisesRegex(TranscriptionError, "invalid stream metadata"):
            self.processor.probe(self.source)

    def _fake_ffmpeg(self, frames, sample_rate=16000, pcm=None):
        def fake_run(command, **kwargs):
            with wave.open(command[-1], "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm if pcm is not None else b"\0\0" * frames)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        return fake_run

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_prepare_without_detector_writes_contiguous_fixed_chunks(self, run):
        run.side_effect = self._fake_ffmpeg(24000)
        destination = self.root / "chunks"
        prepared = self.processor.prepare(
            self.source,
            destination,
            sample_rate=16000,
            segmentation=SegmentationSettings(detector="none", max_chunk_seconds=1),
            max_duration=1.5,
        )
        self.assertEqual(
            [(chunk.start, chunk.end, chunk.index) for chunk in prepared.chunks],
            [(0.0, 1.0, 0), (1.0, 1.5, 1)],
        )
        self.assertEqual(prepared.duration, 1.5)
        self.assertEqual(prepared.speech_seconds, 1.5)
        self.assertEqual(prepared.segmentation["detector"], "none")
        self.assertEqual(prepared.segmentation["settings"]["max_chunk_seconds"], 1)
        json.dumps(prepared.segmentation)
        self.assertTrue(all(chunk.path.is_file() for chunk in prepared.chunks))
        self.assertEqual(
            sorted(path.name for path in destination.iterdir()),
            ["chunk_000000.wav", "chunk_000001.wav"],
        )
        command = run.call_args.args[0]
        self.assertIn("atrim=end_sample=24000", command[command.index("-af") + 1])
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_prepare_slices_detected_regions_and_removes_intermediate(self, run):
        run.side_effect = self._fake_ffmpeg(16000 * 10)

        class Detector:
            name = "fake"

            def check(self):
                pass

            def detect(self, wav_path):
                probabilities = [0.0] * 100
                probabilities[20:40] = [0.9] * 20
                return VoiceActivity(0.1, tuple(probabilities), 0.5, "fake", {"k": 1})

        processor = FFmpegProcessor("ffmpeg-test", "ffprobe-test", detector=Detector())
        destination = self.root / "chunks"
        settings = SegmentationSettings(detector="fake", speech_pad_seconds=0.2)
        prepared = processor.prepare(
            self.source, destination, sample_rate=16000, segmentation=settings
        )
        self.assertEqual(len(prepared.chunks), 1)
        chunk = prepared.chunks[0]
        self.assertAlmostEqual(chunk.start, 1.8)
        self.assertAlmostEqual(chunk.end, 4.2)
        self.assertAlmostEqual(prepared.speech_seconds, 2.4)
        self.assertAlmostEqual(prepared.duration, 10.0)
        with wave.open(str(chunk.path), "rb") as wav:
            self.assertEqual(wav.getnframes(), round(2.4 * 16000))
        self.assertEqual([path.name for path in destination.iterdir()], ["chunk_000000.wav"])
        self.assertEqual(prepared.segmentation["regions"], 1)
        self.assertEqual(prepared.segmentation["details"], {"k": 1})
        json.dumps(prepared.segmentation)

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_prepare_without_speech_returns_no_chunks(self, run):
        run.side_effect = self._fake_ffmpeg(16000)

        class Detector:
            name = "silero"

            def check(self):
                pass

            def detect(self, wav_path):
                return VoiceActivity(0.1, (0.0,) * 10, 0.5, "silero")

        processor = FFmpegProcessor("ffmpeg-test", "ffprobe-test", detector=Detector())
        prepared = processor.prepare(
            self.source,
            self.root / "chunks",
            sample_rate=16000,
            segmentation=SegmentationSettings(),
        )
        self.assertEqual(prepared.chunks, ())
        self.assertEqual(prepared.speech_seconds, 0)
        self.assertEqual(prepared.duration, 1.0)

    def test_prepare_requires_matching_detector(self):
        with self.assertRaisesRegex(TranscriptionError, "not available"):
            self.processor.prepare(
                self.source,
                self.root / "chunks",
                sample_rate=16000,
                segmentation=SegmentationSettings(detector="silero"),
            )

    def test_split_rejects_short_chunks(self):
        path = self.root / "short.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 16000)
        chunk = AudioChunk(path=path, start=0.0, end=1.0, index=3)
        with self.assertRaisesRegex(TranscriptionError, "shorter"):
            self.processor.split(chunk, self.root)

    def test_split_cuts_at_quietest_interior_point_and_nests(self):
        rate = 16000
        loud = struct.pack("<h", 8000) * rate
        quiet = b"\0\0" * round(0.1 * rate)
        pcm = loud * 2 + quiet + loud * 2  # 4.1 s, pause centred near 2.0-2.1 s
        path = self.root / "chunk_000007.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm)
        chunk = AudioChunk(path=path, start=10.0, end=14.1, index=7)
        first, second = self.processor.split(chunk, self.root / "parts")
        self.assertEqual((first.index, first.part, second.part), (7, "a", "b"))
        self.assertEqual(first.start, 10.0)
        self.assertEqual(second.end, 14.1)
        self.assertEqual(first.end, second.start)
        self.assertGreater(first.end, 12.0)
        self.assertLess(first.end, 12.1)
        self.assertEqual(first.path.name, "chunk_000007a.wav")
        self.assertEqual(second.path.name, "chunk_000007b.wav")
        with wave.open(str(first.path), "rb") as one, wave.open(str(second.path), "rb") as two:
            self.assertEqual(one.getnframes() + two.getnframes(), len(pcm) // 2)
        nested = self.processor.split(first, self.root / "parts")
        self.assertEqual([item.part for item in nested], ["aa", "ab"])
        self.assertEqual(nested[0].path.name, "chunk_000007aa.wav")

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_ffmpeg_failure_reports_diagnostic(self, run):
        run.return_value = types.SimpleNamespace(returncode=1, stdout="", stderr="decoder failed")
        with self.assertRaisesRegex(TranscriptionError, "decoder failed"):
            self.processor.probe(self.source)


class MockBackendTests(unittest.TestCase):
    def test_emits_explicit_synthetic_diagnostic(self):
        backend = MockBackend()
        segment = backend.transcribe(AudioChunk(Path("x.wav"), 1, 2, 0), language=None)[0]
        self.assertIn("MOCK TRANSCRIPT", segment.text)
        self.assertIn("no speech recognition performed", segment.text)
        self.assertIsNone(segment.speaker)
        backend.check()
        backend.close()


if __name__ == "__main__":
    unittest.main()
