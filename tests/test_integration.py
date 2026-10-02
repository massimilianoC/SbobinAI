"""Optional real FFmpeg integration on generated public-safe fixtures."""

import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from audio_transcript.adapters.ffmpeg import FFmpegProcessor


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not on PATH")
class MediaIntegrationTests(unittest.TestCase):
    def test_exact_audio_chunks_preserve_all_decoded_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "generated audio.wav"
            with wave.open(str(source), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\0\0" * 40000)
            processor = FFmpegProcessor()
            info = processor.probe(source)
            self.assertAlmostEqual(info.duration, 2.5)
            chunks = processor.prepare(source, root / "chunks", chunk_seconds=1, sample_rate=16000)
            self.assertEqual(len(chunks), 3)
            self.assertTrue(all(chunk.end - chunk.start <= 1 for chunk in chunks))
            self.assertAlmostEqual(chunks[-1].end, 2.5)
            self.assertEqual([chunk.start for chunk in chunks], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
