"""CLI integration of monitoring: flags, config, events command, run files, fingerprint."""

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application.eventlog import events_path, follow_events
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.cli import main
from audio_transcript.config import load_config

try:
    from .fakes import FakeProcessor, make_config
    from .test_events import SentinelBackend
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeProcessor, make_config
    from test_events import SentinelBackend


def run_cli(root: Path, *extra: str, stdout=None, stderr=None, tty=False):
    (root / "input").mkdir(exist_ok=True)
    media = root / "input" / "talk.wav"
    if not media.exists():
        media.write_bytes(b"synthetic media bytes")
    argv = [
        "run",
        "--backend",
        "mock",
        "--vad",
        "none",
        "--input-dir",
        str(root / "input"),
        "--process-dir",
        str(root / "process"),
        "--output-dir",
        str(root / "output"),
        "--processed-dir",
        str(root / "processed"),
        "--no-monitor",
        *extra,
    ]
    out = stdout or io.StringIO()
    err = stderr or io.StringIO()
    with (
        patch("audio_transcript.cli._make_processor", lambda config: FakeProcessor(3)),
        patch("audio_transcript.cli._make_backend", lambda config: SentinelBackend()),
        patch("audio_transcript.cli._make_exporter", FileExporter),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = load_config()
        self.assertEqual((config.ui, config.monitor, config.monitor_interval), ("auto", True, 1.0))

    def test_cli_flags_override_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = {}

            class FakePipeline:
                def __init__(self, config, *args, **kwargs):
                    captured["config"] = config

                def run(self, sources, limit=None):
                    return []

            with (
                patch("audio_transcript.cli.TranscriptionPipeline", FakePipeline),
                patch("audio_transcript.cli._make_processor"),
                patch("audio_transcript.cli._make_backend"),
                patch("audio_transcript.cli._make_exporter"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(
                    [
                        "run",
                        "--backend",
                        "mock",
                        "--input-dir",
                        str(root / "input"),
                        "--process-dir",
                        str(root / "process"),
                        "--output-dir",
                        str(root / "output"),
                        "--processed-dir",
                        str(root / "processed"),
                        "--ui",
                        "jsonl",
                        "--no-monitor",
                        "--monitor-interval",
                        "2.5",
                    ]
                )
            config = captured["config"]
            self.assertEqual(config.ui, "jsonl")
            self.assertFalse(config.monitor)
            self.assertEqual(config.monitor_interval, 2.5)

    def test_invalid_values_are_rejected(self):
        for overrides in (
            {"ui": "fancy"},
            {"monitor_interval": 0.0},
            {"monitor_interval": float("nan")},
            {"monitor_interval": 99999.0},
            {"monitor": "yes"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(None, overrides)

    def test_toml_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.toml"
            path.write_text(
                '[audio_transcript]\nui = "plain"\nmonitor = false\nmonitor_interval = 0.5\n',
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(
                (config.ui, config.monitor, config.monitor_interval), ("plain", False, 0.5)
            )

    def test_fingerprint_ignores_presentation_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prints = set()
            for extra in (
                {},
                {"ui": "live"},
                {"ui": "jsonl", "monitor": False},
                {"monitor_interval": 5.0},
            ):
                config = make_config(root, **extra)
                pipeline = TranscriptionPipeline(config, FakeProcessor(1), None, None)
                prints.add(pipeline._fingerprint())
            self.assertEqual(len(prints), 1)


class CliRunTests(unittest.TestCase):
    def test_plain_mode_prints_exactly_the_status_lines_and_writes_the_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, out, _ = run_cli(root, "--ui", "plain")
            self.assertEqual(code, 0)
            latest = json.loads((root / "process" / "runs" / "latest.json").read_text("utf-8"))
            run_dir = root / "process" / "runs" / latest["run_id"]
            logged = [
                line.split("] ", 1)[1]
                for line in (run_dir / "pipeline.log").read_text("utf-8").splitlines()
                if " [info] " in line or " [warning] " in line or " [error] " in line
            ]
            self.assertEqual(out.splitlines(), logged)
            self.assertIn("Completed talk.wav", out)
            self.assertNotIn("\x1b", out)
            self.assertEqual(latest["state"], "finished")

    def test_auto_is_plain_when_stdout_is_not_a_tty(self):
        with tempfile.TemporaryDirectory() as directory:
            code, out, _ = run_cli(Path(directory), "--ui", "auto")
            self.assertEqual(code, 0)
            self.assertNotIn("\x1b", out)
            self.assertIn("Prepared talk.wav", out)

    def test_jsonl_mode_streams_events_on_stdout_only(self):
        with tempfile.TemporaryDirectory() as directory:
            code, out, _ = run_cli(Path(directory), "--ui", "jsonl")
            self.assertEqual(code, 0)
            records = [json.loads(line) for line in out.splitlines()]
            self.assertEqual(records[0]["type"], "run.started")
            self.assertEqual(records[-1]["type"], "run.finished")
            self.assertEqual(records[-1]["data"]["exit_status"], 0)

    def test_events_command_prints_filters_and_follows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_cli(root, "--ui", "plain")
            base = ["events", "--process-dir", str(root / "process")]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(base), 0)
            all_types = [json.loads(line)["type"] for line in stdout.getvalue().splitlines()]
            self.assertEqual(all_types[0], "run.started")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main([*base, "--type", "chunk."]), 0)
            kinds = {json.loads(line)["type"] for line in stdout.getvalue().splitlines()}
            self.assertEqual(kinds, {"chunk.started", "chunk.attempt", "chunk.finished"})
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main([*base, "--follow", "--poll-interval", "0.05"]), 0)
            self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1])["type"], "run.finished")

    def test_events_command_reports_a_missing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(["events", "--process-dir", str(Path(directory) / "process")])
            self.assertEqual(code, 1)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(["events", "--run", "../x", "--process-dir", directory])
            self.assertEqual(code, 1)


class FollowTests(unittest.TestCase):
    def test_follow_waits_for_lines_and_stops_on_run_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            out: list[str] = []

            def writer():
                time.sleep(0.15)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write('{"type":"run.started","seq":1}\n')
                    stream.write('{"type":"progress","seq":2}\n{"type":"par')
                    stream.flush()
                    time.sleep(0.15)
                    stream.write('tial","seq":3}\n')
                    stream.write('{"type":"run.finished","seq":4}\n{"type":"late","seq":5}\n')

            thread = threading.Thread(target=writer)
            thread.start()
            code = follow_events(path, out.append, follow=True, poll_interval=0.02, max_polls=500)
            thread.join()
            self.assertEqual(code, 0)
            kinds = [json.loads(line)["type"] for line in out]
            self.assertEqual(kinds[:4], ["run.started", "progress", "partial", "run.finished"])

    def test_type_filter_still_stops_on_run_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(
                '{"type":"log","seq":1}\n{"type":"run.finished","seq":2}\n', encoding="utf-8"
            )
            out: list[str] = []
            code = follow_events(
                path, out.append, type_prefix="log", follow=True, poll_interval=0.01, max_polls=3
            )
            self.assertEqual((code, len(out)), (0, 1))

    def test_missing_file_without_follow_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(follow_events(Path(directory) / "nope.jsonl", print), 1)
            self.assertIsNone(events_path(Path(directory), "latest"))


if __name__ == "__main__":
    unittest.main()


class WizardSessionTests(unittest.TestCase):
    def test_wizard_processing_phase_uses_the_same_event_bus(self):
        from audio_transcript.application.session import RunSession
        from audio_transcript.application.wizard import Wizard

        try:
            from .fakes import FakeBackend
        except ImportError:  # discovered with ``-s tests``
            from fakes import FakeBackend

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            (root / "input" / "a.wav").write_bytes(b"synthetic")
            config = make_config(root, backend="llamacpp", model="fake", response_mode="qwen3-asr")
            lines: list[str] = []
            sessions: list[RunSession] = []

            def make(cfg):
                return TranscriptionPipeline(
                    cfg, FakeProcessor(2), FakeBackend(), FileExporter(), status=lines.append
                )

            def open_session(queue_size):
                session = RunSession(
                    cfg_holder[0], command="wizard", queue_size=queue_size, status_out=lines.append
                )
                sessions.append(session)
                return session

            cfg_holder = [config]
            replies = iter(["", "", "=", "", "y"])
            wizard = Wizard(
                config,
                make,
                ask=lambda prompt: next(replies, ""),
                say=lines.append,
                last_answers_path=None,
                session_factory=open_session,
            )
            self.assertEqual(wizard.run(), 0)
            (session,) = sessions
            events = [
                json.loads(line)
                for line in session.paths.events.read_text(encoding="utf-8").splitlines()
            ]
            kinds = [e["type"] for e in events]
            self.assertEqual(kinds[0], "run.started")
            self.assertEqual(events[0]["data"]["command"], "wizard")
            self.assertIn("job.finished", kinds)
            self.assertEqual(kinds[-1], "run.finished")
            self.assertTrue(any(line.startswith("Completed a.wav") for line in lines))


class LiveRunTests(unittest.TestCase):
    def test_live_run_draws_frames_restores_the_cursor_and_prints_a_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("audio_transcript.ui.resolve_ui_mode", lambda requested: "live"):
                code, out, _ = run_cli(Path(directory), "--ui", "live")
            self.assertEqual(code, 0)
            self.assertTrue(out.startswith("\x1b[?25l"))
            self.assertIn("\x1b[?25h", out)
            self.assertIn("Chunks [", out)
            self.assertIn("Run summary", out)
            self.assertIn("completed | talk.wav", out)
            # the status lines are shown inside the display, not printed one by one
            self.assertNotIn("\nCompleted talk.wav: 3 segments\n", out)
