"""Check that the central sampling utility rejects silence and accepts signal."""

import contextlib
import importlib.util
import io
import json
import math
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not on PATH")
class SamplingTests(unittest.TestCase):
    def test_silence_rejection_and_central_signal_selection(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "sample-audio.py"
        spec = importlib.util.spec_from_file_location("sample_audio", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for has_signal in (False, True):
                with self.subTest(signal=has_signal):
                    source = root / ("tone.wav" if has_signal else "silence.wav")
                    with wave.open(str(source), "wb") as stream:
                        stream.setnchannels(1)
                        stream.setsampwidth(2)
                        stream.setframerate(16000)
                        samples = (
                            int(10000 * math.sin(2 * math.pi * 440 * index / 16000))
                            if has_signal
                            else 0
                            for index in range(32000)
                        )
                        stream.writeframes(
                            b"".join(struct.pack("<h", sample) for sample in samples)
                        )
                    destination = root / source.stem
                    argv = [
                        str(script),
                        str(source),
                        "--destination",
                        str(destination),
                        "--count",
                        "1",
                        "--attempts",
                        "1",
                        "--seconds",
                        "0.5",
                        "--seed",
                        "7",
                    ]
                    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                        code = module.main()
                    self.assertEqual(code, 0 if has_signal else 1)
                    report = json.loads(
                        (destination / "sampling-report.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(report["checked"][0]["has_signal"], has_signal)
                    start = report["checked"][0]["start_seconds"]
                    self.assertGreaterEqual(start, 0.7)
                    self.assertLessEqual(start, 1.3)
                    self.assertEqual(len(list(destination.glob("*.wav"))), 1 if has_signal else 0)


if __name__ == "__main__":
    unittest.main()
