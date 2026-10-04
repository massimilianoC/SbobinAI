"""Exit statuses and JSON shapes through real processes (python -m and the console scripts)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_module(*args, stdin=None, cwd=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    return subprocess.run(
        [sys.executable, "-m", "audio_transcript", *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd or ROOT,
        timeout=120,
    )


def console_script(name):
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    return str(candidate) if candidate.is_file() else None


class SubprocessExitTests(unittest.TestCase):
    def test_success_statuses(self):
        for args in (["--version"], ["--help"], ["help", "run"], ["man", "doctor"]):
            result = run_module(*args)
            self.assertEqual(result.returncode, 0, (args, result.stderr))
            self.assertEqual(result.stderr, "")
        self.assertIn("sbobinai", run_module("--version").stdout)

    def test_help_equals_dash_help(self):
        self.assertEqual(run_module("help", "events").stdout, run_module("events", "--help").stdout)

    def test_usage_error_status_and_streams(self):
        result = run_module("run", "--no-such-option")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("unrecognized arguments", result.stderr)

    def test_usage_error_json(self):
        result = run_module("run", "--no-such-option", "--json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr, "")
        error = json.loads(result.stdout)["error"]
        self.assertEqual((error["code"], error["exit_status"]), ("usage", 2))

    def test_config_error_text_and_json(self):
        text = run_module("run", "--config", "no-such-file.toml")
        self.assertEqual(text.returncode, 2)
        self.assertIn("ERROR:", text.stderr)
        self.assertEqual(text.stdout, "")
        as_json = run_module("doctor", "--config", "no-such-file.toml", "--json")
        self.assertEqual(as_json.returncode, 2)
        self.assertEqual(as_json.stderr, "")
        data = json.loads(as_json.stdout)
        self.assertEqual(data["error"]["code"], "config_invalid")
        self.assertFalse(data["ok"])

    def test_runtime_error_json(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_module("events", "--process-dir", directory, "--json")
        self.assertEqual(result.returncode, 1)
        data = json.loads(result.stdout)
        self.assertEqual(data["error"]["code"], "run_not_found")
        self.assertEqual(data["error"]["exit_status"], 1)

    def test_man_json_matches_in_process_output(self):
        from audio_transcript import climan

        result = run_module("man", "--format", "json")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, climan.json_text())

    def test_empty_queue_run_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_module(
                "run",
                "--prepare-only",
                "--vad",
                "none",
                "--no-monitor",
                "--json",
                "--input-dir",
                str(root / "i"),
                "--process-dir",
                str(root / "p"),
                "--output-dir",
                str(root / "o"),
                "--processed-dir",
                str(root / "d"),
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        last = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(last["type"], "result")
        self.assertTrue(last["ok"])
        self.assertEqual(last["jobs"], [])

    def test_doctor_json_shape(self):
        result = run_module("doctor", "--backend", "mock", "--vad", "none", "--json")
        self.assertIn(result.returncode, (0, 1))
        data = json.loads(result.stdout)
        self.assertEqual(data["ok"], result.returncode == 0)
        self.assertEqual([c["name"] for c in data["checks"]][-2:], ["speech_detector", "backend"])
        self.assertEqual(result.stderr, "")

    def test_repl_scripted_session(self):
        result = run_module("repl", stdin="help\nman run\nexit\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn("SbobinAI", result.stdout)
        self.assertIn("Exit status", result.stdout)

    def test_console_script_aliases(self):
        found = False
        for name in ("sbobinai", "audio-transcript", "cli-anything-sbobinai"):
            script = console_script(name)
            if script is None:
                continue
            found = True
            result = subprocess.run(
                [script, "--version"], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(result.returncode, 0, name)
            self.assertIn("sbobinai", result.stdout)
        if not found:
            self.skipTest("console scripts are not installed in this environment")


if __name__ == "__main__":
    unittest.main()
