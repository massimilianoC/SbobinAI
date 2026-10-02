"""Optional real FFmpeg integration on generated public-safe fixtures."""

import math
import random
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from audio_transcript.adapters.ffmpeg import FFmpegProcessor
from audio_transcript.adapters.vad import EnergyDetector
from audio_transcript.domain.models import SegmentationSettings

RATE = 16000


def write_wav(path, samples):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def tone(seconds, amplitude=8000):
    return [
        int(amplitude * math.sin(2 * math.pi * 220 * i / RATE)) for i in range(int(seconds * RATE))
    ]


def hiss(seconds, seed):
    rng = random.Random(seed)
    return [rng.randint(-15, 15) for _ in range(int(seconds * RATE))]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not on PATH")
class MediaIntegrationTests(unittest.TestCase):
    def test_fixed_chunks_preserve_all_decoded_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "generated audio.wav"
            write_wav(source, [0] * 40000)
            processor = FFmpegProcessor()
            info = processor.probe(source)
            self.assertAlmostEqual(info.duration, 2.5)
            prepared = processor.prepare(
                source,
                root / "chunks",
                sample_rate=RATE,
                segmentation=SegmentationSettings(detector="none", max_chunk_seconds=1),
            )
            chunks = prepared.chunks
            self.assertEqual(len(chunks), 3)
            self.assertTrue(all(chunk.end - chunk.start <= 1 for chunk in chunks))
            self.assertAlmostEqual(chunks[-1].end, 2.5)
            self.assertEqual([chunk.start for chunk in chunks], [0, 1, 2])
            self.assertAlmostEqual(prepared.duration, 2.5)
            self.assertAlmostEqual(prepared.speech_seconds, 2.5)

    def test_max_duration_bounds_the_analysed_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long.wav"
            write_wav(source, [0] * 5 * RATE)
            prepared = FFmpegProcessor().prepare(
                source,
                root / "chunks",
                sample_rate=RATE,
                segmentation=SegmentationSettings(detector="none", max_chunk_seconds=2),
                max_duration=3,
            )
            self.assertAlmostEqual(prepared.duration, 3.0)
            self.assertEqual(len(prepared.chunks), 2)

    def test_energy_segmentation_keeps_speech_and_drops_silence(self):
        samples = hiss(3, 1) + tone(2) + hiss(4, 2) + tone(3) + hiss(3, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speechlike.wav"
            write_wav(source, samples)
            processor = FFmpegProcessor(detector=EnergyDetector())
            prepared = processor.prepare(
                source,
                root / "chunks",
                sample_rate=RATE,
                segmentation=SegmentationSettings(detector="energy", max_chunk_seconds=15),
            )
            self.assertAlmostEqual(prepared.duration, 15.0)
            self.assertEqual(len(prepared.chunks), 2)
            first, second = prepared.chunks
            self.assertAlmostEqual(first.start, 2.8, delta=0.15)
            self.assertAlmostEqual(first.end, 5.2, delta=0.15)
            self.assertAlmostEqual(second.start, 8.8, delta=0.15)
            self.assertAlmostEqual(second.end, 12.2, delta=0.15)
            self.assertAlmostEqual(prepared.speech_seconds, 5.8, delta=0.3)
            self.assertEqual(
                sorted(p.name for p in (root / "chunks").iterdir()),
                [
                    "chunk_000000.wav",
                    "chunk_000001.wav",
                ],
            )
            self.assertEqual(prepared.segmentation["detector"], "energy")
            self.assertEqual(prepared.segmentation["regions"], 2)

    def test_split_of_a_real_chunk_covers_the_parent(self):
        samples = tone(2) + [0] * (RATE // 10) + tone(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pause.wav"
            write_wav(source, samples)
            processor = FFmpegProcessor()
            prepared = processor.prepare(
                source,
                root / "chunks",
                sample_rate=RATE,
                segmentation=SegmentationSettings(detector="none", max_chunk_seconds=15),
            )
            parent = prepared.chunks[0]
            first, second = processor.split(parent, root / "parts")
            self.assertEqual((first.start, second.end), (parent.start, parent.end))
            self.assertAlmostEqual(first.end, 2.05, delta=0.05)


if __name__ == "__main__":
    unittest.main()
