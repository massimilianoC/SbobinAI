"""Atomic replace must survive brief Windows sharing locks (antivirus, readers)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript import fsutil
from audio_transcript.application.state import atomic_json


class ReplaceWithRetryTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.sleeps = []
        patcher = patch.object(fsutil.time, "sleep", self.sleeps.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def flaky_replace(self, failures):
        real = fsutil.os.replace
        state = {"calls": 0}

        def replace(source, destination):
            state["calls"] += 1
            if state["calls"] <= failures:
                raise PermissionError(5, "Access is denied")
            real(source, destination)

        return replace, state

    def test_transient_permission_errors_are_retried(self):
        source, destination = self.root / "a.tmp", self.root / "a.json"
        source.write_text("new", encoding="utf-8")
        destination.write_text("old", encoding="utf-8")
        replace, state = self.flaky_replace(failures=3)
        with patch.object(fsutil.os, "replace", replace):
            fsutil.replace_with_retry(source, destination)
        self.assertEqual(destination.read_text(encoding="utf-8"), "new")
        self.assertEqual(state["calls"], 4)
        self.assertEqual(len(self.sleeps), 3)
        self.assertEqual(self.sleeps, sorted(self.sleeps))  # backoff never shrinks

    def test_persistent_lock_is_raised_after_the_last_attempt(self):
        replace, state = self.flaky_replace(failures=100)
        with patch.object(fsutil.os, "replace", replace):
            with self.assertRaises(PermissionError):
                fsutil.replace_with_retry(self.root / "x.tmp", self.root / "x.json")
        self.assertEqual(state["calls"], fsutil.REPLACE_ATTEMPTS)
        self.assertLessEqual(sum(self.sleeps), 3.0)

    def test_other_errors_are_not_retried(self):
        with self.assertRaises(FileNotFoundError):
            fsutil.replace_with_retry(self.root / "missing.tmp", self.root / "x.json")
        self.assertEqual(self.sleeps, [])

    def test_checkpoint_write_survives_a_brief_lock(self):
        path = self.root / "checkpoint.json"
        atomic_json(path, {"chunks": {"0": []}})
        replace, _ = self.flaky_replace(failures=2)
        with patch.object(fsutil.os, "replace", replace):
            atomic_json(path, {"chunks": {"0": [], "1": []}})
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")), {"chunks": {"0": [], "1": []}}
        )
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["checkpoint.json"])


if __name__ == "__main__":
    unittest.main()
