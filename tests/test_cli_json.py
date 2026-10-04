"""--json mode, structured errors, doctor checks and the interactive shell."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript import cli, clidoc
from audio_transcript.cli import _doctor, main
from audio_transcript.config import load_config

try:
    from .test_monitoring_cli import run_cli
except ImportError:  # discovered with ``-s tests``
    from test_monitoring_cli import run_cli


def run_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
    return code, out.getvalue(), err.getvalue()


def only_json(text):
    return json.loads(text.strip().splitlines()[-1])


class FakeBackend:
    def __init__(self, fail=False):
        self.fail = fail

    def check(self):
        if self.fail:
            raise RuntimeError("server not reachable")

    def close(self):
        return None


class FakeDetector:
    def check(self):
        return None


def patched_checks(*, backend_fails=False, ffmpeg=True):
    which = (lambda name: f"/bin/{name}") if ffmpeg else (lambda name: None)
    completed = type("Completed", (), {"returncode": 0, "stderr": ""})()
    return (
        patch("audio_transcript.cli.shutil.which", which),
        patch("audio_transcript.cli.subprocess.run", lambda *a, **k: completed),
        patch("audio_transcript.cli._make_detector", return_value=FakeDetector()),
        patch("audio_transcript.cli._make_backend", return_value=FakeBackend(backend_fails)),
    )


class DoctorJsonTests(unittest.TestCase):
    def doctor(self, as_json, **kwargs):
        config = load_config(None, {"backend": "mock", "vad": "energy"})
        out, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            for patcher in patched_checks(**kwargs):
                stack.enter_context(patcher)
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            code = _doctor(config, as_json)
        return code, out.getvalue(), err.getvalue()

    def test_ready(self):
        code, out, err = self.doctor(True)
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(data["ok"])
        self.assertEqual(
            [c["name"] for c in data["checks"]],
            ["ffmpeg", "ffprobe", "speech_detector", "backend"],
        )
        for check in data["checks"]:
            self.assertEqual(set(check), {"name", "ok", "detail"})
            self.assertTrue(check["ok"])
        self.assertEqual(err, "")
        self.assertNotIn("error", data)

    def test_failing_backend(self):
        code, out, err = self.doctor(True, backend_fails=True)
        data = json.loads(out)
        self.assertEqual(code, 1)
        self.assertFalse(data["ok"])
        failed = [c for c in data["checks"] if not c["ok"]]
        self.assertEqual([c["name"] for c in failed], ["backend"])
        self.assertIn("server not reachable", failed[0]["detail"])
        self.assertEqual(data["error"]["code"], "checks_failed")
        self.assertEqual(err, "")

    def test_missing_ffmpeg(self):
        with patch("audio_transcript.cli.Path.is_file", return_value=False):
            code, out, _ = self.doctor(True, ffmpeg=False)
        data = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual([c["name"] for c in data["checks"] if not c["ok"]], ["ffmpeg", "ffprobe"])

    def test_text_output_is_unchanged(self):
        code, out, err = self.doctor(False)
        self.assertEqual(code, 0)
        self.assertIn("Speech detector available: energy", out)
        self.assertIn("Backend available: mock (mock)", out)
        self.assertIn("Media tools available: ffmpeg, ffprobe", out)
        code, out, err = self.doctor(False, backend_fails=True)
        self.assertEqual(code, 1)
        self.assertIn("ERROR: Backend unavailable (mock): RuntimeError: server not reachable", err)
        self.assertNotIn("Media tools available", out)


class StructuredErrorTests(unittest.TestCase):
    def assert_error(self, argv, code, status):
        exit_code, out, err = run_main(*argv)
        self.assertEqual(exit_code, status, out)
        data = json.loads(out)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], code)
        self.assertEqual(data["error"]["exit_status"], status)
        self.assertTrue(data["error"]["message"])
        self.assertEqual(err, "")

    def test_config_invalid(self):
        self.assert_error(["run", "--config", "missing.toml", "--json"], "config_invalid", 2)
        self.assert_error(["doctor", "--config", "missing.toml", "--json"], "config_invalid", 2)

    def test_usage_error_is_json(self):
        self.assert_error(["run", "--no-such-option", "--json"], "usage", 2)
        self.assert_error(["help", "nope", "--json"], "usage", 2)

    def test_usage_error_is_text_without_json(self):
        code, out, err = run_main("run", "--no-such-option")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("unrecognized arguments", err)

    def test_unreadable_prompt_file(self):
        self.assert_error(["run", "--prompt-file", "missing.txt", "--json"], "usage", 2)

    def test_events_run_not_found(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assert_error(["events", "--process-dir", directory, "--json"], "run_not_found", 1)
            self.assert_error(
                ["events", "--run", "nope", "--process-dir", directory, "--json"],
                "events_missing",
                1,
            )

    def test_server_profile_without_server_table(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "c.toml"
            config.write_text('[audio_transcript]\nbackend = "mock"\n', encoding="utf-8")
            self.assert_error(
                ["server-profile", "--config", str(config), "--json"], "server_not_configured", 2
            )
            code, out, _ = run_main("server-profile", "--config", str(config), "--optional")
            self.assertEqual((code, out.strip()), (0, "null"))

    def test_wizard_json_requires_answers(self):
        self.assert_error(["wizard", "--backend", "mock", "--json"], "interactive_required", 2)

    def test_init_config_exists_and_setup_needs_yes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.toml"
            target.write_text("x", encoding="utf-8")
            self.assert_error(
                ["init-config", "--store", directory, "--output", str(target), "--json"],
                "config_exists",
                1,
            )
            code, out, _ = run_main(
                "init-config",
                "--store",
                directory,
                "--output",
                str(Path(directory) / "new.toml"),
                "--json",
            )
            data = json.loads(out)
            self.assertEqual((code, data["ok"], data["command"]), (0, True, "init-config"))
            self.assertTrue((Path(directory) / "new.toml").is_file())

    def test_setup_dry_run_json(self):
        with tempfile.TemporaryDirectory() as directory:
            code, out, err = run_main("setup", "--store", directory, "--dry-run", "--json")
            data = json.loads(out)
            self.assertEqual(code, 0, out)
            self.assertTrue(data["ok"])
            self.assertTrue(data["dry_run"])
            self.assertTrue(data["plan"])
            self.assertEqual(err, "")


class OneShotJsonTests(unittest.TestCase):
    def test_catalog_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, out, err = run_main(
                "catalog",
                "--process-dir",
                str(root / "process"),
                "--output-dir",
                str(root / "output"),
                "--input-dir",
                str(root / "input"),
                "--processed-dir",
                str(root / "processed"),
                "--json",
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(
                json.loads(out), {"ok": True, "command": "catalog", "sources": 0, "versions": 0}
            )

    def test_migrate_layout_json_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, out, _ = run_main(
                "migrate-layout",
                "--process-dir",
                str(root / "process"),
                "--output-dir",
                str(root / "output"),
                "--input-dir",
                str(root / "input"),
                "--processed-dir",
                str(root / "processed"),
                "--json",
            )
            data = json.loads(out)
            self.assertEqual(code, 0)
            self.assertTrue(data["dry_run"])
            self.assertEqual(data["to_migrate"], 0)

    def test_help_json(self):
        code, out, _ = run_main("help", "run", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["commands"][0]["name"], "run")


class RunJsonTests(unittest.TestCase):
    def test_run_json_streams_events_and_ends_with_result(self):
        with tempfile.TemporaryDirectory() as directory:
            code, out, err = run_cli(Path(directory), "--json")
            lines = [json.loads(line) for line in out.strip().splitlines()]
            self.assertEqual(code, 0, err)
            self.assertEqual(lines[0]["type"], "run.started")
            self.assertEqual(lines[0]["data"]["ui"], "jsonl")
            result = lines[-1]
            self.assertEqual(result["type"], "result")
            self.assertTrue(result["ok"])
            self.assertEqual(result["exit_status"], 0)
            self.assertEqual(result["command"], "run")
            job = result["jobs"][0]
            self.assertEqual(job["status"], "completed")
            self.assertTrue(job["transcript"].endswith("transcript.txt"))
            self.assertTrue(Path(job["transcript"]).is_file())
            # the second run is idempotent: the finished version is skipped
            code, out, _ = run_cli(Path(directory), "--json")
            again = only_json(out)
            self.assertEqual(code, 0)
            self.assertEqual(again["jobs"][0]["status"], "skipped")

    def test_run_json_failed_job_carries_error(self):
        results = [{"job_id": "a/b", "source": "x.wav", "status": "failed", "error": "boom"}]

        class FakePipeline:
            def __init__(self, config, *args, **kwargs):
                pass

            def run(self, sources, limit=None):
                return results

        out = io.StringIO()
        with (
            patch("audio_transcript.cli.TranscriptionPipeline", FakePipeline),
            patch("audio_transcript.cli._make_processor"),
            patch("audio_transcript.cli._make_backend"),
            patch("audio_transcript.cli._make_exporter"),
            contextlib.redirect_stdout(out),
        ):
            with tempfile.TemporaryDirectory() as directory:
                code = main(
                    [
                        "run",
                        "--backend",
                        "mock",
                        "--no-monitor",
                        "--json",
                        "--process-dir",
                        directory + "/p",
                        "--output-dir",
                        directory + "/o",
                        "--input-dir",
                        directory + "/i",
                        "--processed-dir",
                        directory + "/d",
                    ]
                )
        result = only_json(out.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "job_failed")
        self.assertEqual(result["jobs"][0]["error"], "boom")

    def test_backend_failure_is_classified(self):
        out = io.StringIO()
        with (
            patch("audio_transcript.cli._make_processor"),
            patch("audio_transcript.cli._make_backend", side_effect=RuntimeError("no runtime")),
            contextlib.redirect_stdout(out),
            tempfile.TemporaryDirectory() as directory,
        ):
            code = main(
                [
                    "run",
                    "--backend",
                    "mock",
                    "--json",
                    "--process-dir",
                    directory + "/p",
                    "--output-dir",
                    directory + "/o",
                    "--input-dir",
                    directory + "/i",
                    "--processed-dir",
                    directory + "/d",
                ]
            )
        data = json.loads(out.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(data["error"]["code"], "backend_unavailable")

    def test_no_files_json_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, out, err = run_main(
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
            result = only_json(out)
            self.assertEqual((code, result["ok"], result["jobs"]), (0, True, []))
            self.assertEqual(err, "")


class ReplTests(unittest.TestCase):
    def repl(self, lines, *args):
        out, err = io.StringIO(), io.StringIO()
        stdin = io.StringIO("\n".join(lines) + "\n")
        with (
            patch("sys.stdin", stdin),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = main(["repl", *args])
        return code, out.getvalue(), err.getvalue()

    def test_banner_help_and_exit(self):
        code, out, _ = self.repl(["help", "exit"])
        self.assertEqual(code, 0)
        self.assertIn("SbobinAI", out)
        self.assertIn("model :", out)
        self.assertIn("Built-ins:", out)

    def test_end_of_input_exits_cleanly(self):
        code, out, _ = self.repl([])
        self.assertEqual(code, 0)

    def test_commands_remember_config_and_report_status(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "c.toml"
            config.write_text('[audio_transcript]\nbackend = "mock"\nmodel = "shell-model"\n')
            code, out, err = self.repl(
                ["config", "server-profile", "quit"], "--config", str(config)
            )
            self.assertEqual(code, 0)
            self.assertIn("shell-model (mock)", out)
            self.assertIn(f"config: {config}", out)
            self.assertIn("[exit status 2]", out)  # no [server] table
            self.assertIn("no [server] table", err)

    def test_session_config_can_be_set(self):
        code, out, _ = self.repl(["config some.toml", "config", "exit"])
        self.assertIn("config: some.toml", out)

    def test_usage_errors_do_not_end_the_session(self):
        code, out, err = self.repl(["run --nonsense", "help run", "exit"])
        self.assertEqual(code, 0)
        self.assertIn("unrecognized arguments", err)
        self.assertIn("Exit status:", out)

    def test_json_mode_prefixes_commands(self):
        code, out, _ = self.repl(["man run", "exit"], "--json")
        self.assertEqual(code, 0)
        self.assertTrue(out.lstrip().startswith("{"))
        self.assertEqual(json.loads(out)["commands"][0]["name"], "run")

    def test_nested_repl_is_refused(self):
        _, _, err = self.repl(["repl", "exit"])
        self.assertIn("already inside", err)

    def test_split_line_keeps_windows_paths(self):
        from audio_transcript import repl

        with patch.object(repl.os, "name", "nt"):
            self.assertEqual(
                repl.split_line('run --file "input\\my talk.mp4"'),
                ["run", "--file", "input\\my talk.mp4"],
            )
        self.assertIs(cli.COMMAND_NAMES, cli.COMMAND_NAMES)
        self.assertIn("repl", clidoc.COMMAND_BY_NAME)


if __name__ == "__main__":
    unittest.main()
