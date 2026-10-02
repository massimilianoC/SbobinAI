"""CLI error contracts without model downloads or private fixtures."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript.cli import main


class CliTests(unittest.TestCase):
    def test_invalid_config_returns_usage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.toml"
            config.write_text('sample_rate = "invalid"\n', encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["run", "--config", str(config)])
            self.assertEqual(code, 2)
            self.assertIn("sample_rate", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_missing_runtime_fails_before_media_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "generic.wav").touch()
            stderr = io.StringIO()
            with patch("audio_transcript.cli._make_processor") as factory:
                with contextlib.redirect_stderr(stderr):
                    code = main(
                        [
                            "run",
                            "--input-dir",
                            str(inputs),
                            "--process-dir",
                            str(root / "process"),
                            "--output-dir",
                            str(root / "output"),
                        ]
                    )
            self.assertEqual(code, 1)
            factory.return_value.prepare.assert_not_called()
            self.assertIn("Nexa", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
