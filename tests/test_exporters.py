"""Tests for actual output contracts and timestamp validation."""

import json
import tempfile
import unittest
from pathlib import Path

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.domain.models import Segment, Transcript, TranscriptionError


class ExporterTests(unittest.TestCase):
    def test_utf8_and_subtitle_formats(self):
        transcript = Transcript(
            "sample",
            "nexa",
            "qwen2audio",
            "it",
            62.125,
            [
                Segment(0, 30.001, "Caffè & <audio>\n\nSecond line"),
                Segment(30.001, 62.125, "Fine."),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            FileExporter().export(transcript, path)
            self.assertEqual(len(list(path.iterdir())), 6)
            payload = json.loads((path / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["segments"][0]["text"], transcript.segments[0].text)
            self.assertIn("coarse", payload["warnings"][0])
            srt = (path / "transcript.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:30,001 --> 00:01:02,125", srt)
            self.assertNotIn("<audio>\n\nSecond", srt)
            vtt = (path / "transcript.vtt").read_text(encoding="utf-8")
            self.assertTrue(vtt.startswith("WEBVTT\n\n"))
            self.assertIn("Caffè &amp; &lt;audio&gt;", vtt)

    def test_mock_is_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            FileExporter().export(Transcript("test", "mock", "mock", None, 1), path)
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "mock_completed")
            self.assertTrue(any("Synthetic" in item for item in report["warnings"]))

    def test_invalid_times_fail_before_writes(self):
        cases = [
            Segment(-1, 1, "x"),
            Segment(1, 1, "x"),
            Segment(0, 3, "x"),
            Segment(float("nan"), 1, "x"),
            Segment(0, float("inf"), "x"),
            Segment(0, 0.0001, "x"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for segment in cases:
                with self.subTest(segment=segment), self.assertRaises(TranscriptionError):
                    FileExporter().export(
                        Transcript("test", "nexa", "qwen", None, 2, [segment]), path
                    )
            self.assertEqual(list(path.iterdir()), [])

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(TranscriptionError):
            FileExporter().export(
                Transcript(
                    "test",
                    "nexa",
                    "qwen",
                    None,
                    2,
                    [
                        Segment(0, 1, "first"),
                        Segment(0.5, 2, "second"),
                    ],
                ),
                Path(directory),
            )


if __name__ == "__main__":
    unittest.main()
