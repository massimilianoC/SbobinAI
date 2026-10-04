"""Interactive wizard: scripted console, synthetic media, fake processor and backend."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.application.wizard import Wizard, load_last_answers
from audio_transcript.cli import main
from audio_transcript.config import load_config

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeBackend, FakeProcessor, make_config


class Console:
    """Scripted ``input``: records prompts, raises EOFError when the script runs out."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.lines: list[str] = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if not self.replies:
            raise EOFError
        return self.replies.pop(0)

    def say(self, line):
        self.lines.append(line)

    @property
    def text(self):
        return "\n".join(self.lines)


class WizardCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "input").mkdir()
        self.configs: list = []
        self.last = self.root / "last" / "last-answers.json"

    def media(self, *names):
        for name in names:
            (self.root / "input" / name).write_bytes(f"synthetic {name}".encode())

    def config(self, **overrides):
        options = {"backend": "llamacpp", "model": "fake", "response_mode": "qwen3-asr"}
        options.update(overrides)
        return make_config(self.root, **options)

    def factory(self, console):
        def make(config):
            self.configs.append(config)
            return TranscriptionPipeline(
                config, FakeProcessor(2), FakeBackend(), FileExporter(), status=console.say
            )

        return make

    def wizard(self, console, config=None, **kwargs):
        return Wizard(
            config or self.config(),
            self.factory(console),
            ask=console.ask,
            say=console.say,
            last_answers_path=self.last,
            **kwargs,
        )

    def run_configs(self):
        """Configurations used for processing (the first one only discovers the queue)."""
        return self.configs[1:]


class AutoLanguageTests(unittest.TestCase):
    def test_auto_overrides_config_language_and_is_recorded_as_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.toml"
            path.write_text('language = "it"\n', encoding="utf-8")
            self.assertEqual(load_config(path).language, "it")
            self.assertIsNone(load_config(path, {"language": "auto"}).language)
            self.assertIsNone(load_config(path, {"language": " AUTO "}).language)
            path.write_text('language = "Auto"\n', encoding="utf-8")
            self.assertIsNone(load_config(path).language)

    def test_auto_has_the_fingerprint_of_no_language(self):
        root = Path("unused")
        auto = load_config(None, {"language": "auto", "backend": "mock"})
        plain = load_config(None, {"backend": "mock"})
        italian = load_config(None, {"language": "it", "backend": "mock"})

        def fingerprint(config):
            return TranscriptionPipeline(config, FakeProcessor())._fingerprint()

        self.assertEqual(fingerprint(auto), fingerprint(plain))
        self.assertNotEqual(fingerprint(auto), fingerprint(italian))
        self.assertIsNotNone(root)


class WizardFlowTests(WizardCase):
    def test_happy_path_one_file(self):
        self.media("a.wav")
        console = Console(["it", "Names: Rossi, Bianchi", "", "y", ""])
        code = self.wizard(console).run()
        self.assertEqual(code, 0)
        (config,) = self.run_configs()
        self.assertEqual(config.language, "it")
        self.assertEqual(config.prompt, "Names: Rossi, Bianchi")
        self.assertIsNone(config.max_duration)
        self.assertIn("status        : completed", console.text)
        self.assertIn("Queue: 1 file(s)", console.text)
        self.assertTrue(any(Path(self.root / "output").glob("*/*/transcript.txt")))

    def test_scope_sets_max_duration_and_new_version(self):
        self.media("a.wav")
        console = Console(["auto", "", "5", "y"])
        self.assertEqual(self.wizard(console).run(), 0)
        (config,) = self.run_configs()
        self.assertIsNone(config.language)
        self.assertEqual(config.max_duration, 300.0)
        self.assertIn("first 5 min only", console.text)
        self.assertIn("earlier versions are kept", console.text)

    def test_two_files_same_answers(self):
        self.media("a.wav", "b.wav")
        console = Console(["it", "topic X", "", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        first, second = self.run_configs()
        self.assertEqual((first.language, first.prompt), (second.language, second.prompt))
        self.assertEqual(console.text.count("status        : completed"), 2)

    def test_two_files_per_file_answers_have_distinct_fingerprints(self):
        self.media("a.wav", "b.wav")
        console = Console(["it", "context one", "", "n", "it", "context two", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        first, second = self.run_configs()
        self.assertEqual(first.prompt, "context one")
        self.assertEqual(second.prompt, "context two")

        def fingerprint(config):
            return TranscriptionPipeline(config, FakeProcessor())._fingerprint()

        self.assertNotEqual(fingerprint(first), fingerprint(second))
        self.assertIn("Answers for b.wav", console.text)

    def test_context_from_file(self):
        self.media("a.wav")
        glossary = self.root / "glossary.txt"
        glossary.write_text("  Camillucci, Qwen3\n", encoding="utf-8")
        console = Console(["it", f"@{glossary}", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertEqual(self.run_configs()[0].prompt, "Camillucci, Qwen3")

    def test_unreadable_or_oversized_context_is_asked_again(self):
        self.media("a.wav")
        console = Console(["it", f"@{self.root / 'missing.txt'}", "x" * 2001, "ok", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertIn("Could not read the UTF-8 context file", console.text)
        self.assertIn("limit is 2000", console.text)
        self.assertEqual(self.run_configs()[0].prompt, "ok")

    def test_unsupported_language_is_explained_and_asked_again(self):
        self.media("a.wav")
        console = Console(["xx", "it", "", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertIn("not a language the Qwen3-ASR model supports", console.text)
        self.assertEqual(self.run_configs()[0].language, "it")
        self.assertEqual(sum("Spoken language" in p for p in console.prompts), 2)

    def test_language_names_become_the_iso_code(self):
        for typed in ("italiano", "Italian", "IT", "it-IT", " Italiano "):
            with self.subTest(typed=typed):
                self.configs.clear()
                self.media("a.wav")
                console = Console([typed, "", "", "y", ""])
                self.assertEqual(self.wizard(console).run(), 0)
                self.assertEqual(self.run_configs()[0].language, "it")

    def test_transcript_language_is_not_asked(self):
        self.media("a.wav")
        console = Console(["it", "", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertFalse(any("Transcript language" in p for p in console.prompts))
        self.assertIn("written in the spoken language", console.text)
        self.assertIn("Translation is out of scope", console.text)

    def test_json_mode_says_context_replaces_the_instruction(self):
        self.media("a.wav")
        console = Console(["it", "Transcribe verbatim.", "", "y", ""])
        self.wizard(console, self.config(response_mode="json")).run()
        self.assertIn("replaces the built-in instruction prompt", console.text)

    def test_cancel_processes_nothing(self):
        self.media("a.wav")
        console = Console(["it", "", "", "n"])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertEqual(self.run_configs(), [])
        self.assertIn("Cancelled. Nothing was processed.", console.text)
        self.assertFalse((self.root / "output").exists() and any((self.root / "output").iterdir()))
        self.assertEqual(len(list((self.root / "input").iterdir())), 1)

    def test_end_of_input_cancels(self):
        self.media("a.wav")
        console = Console(["it"])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertEqual(self.run_configs(), [])

    def test_empty_queue_explains_how_to_add_files(self):
        console = Console([])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertIn("No media files found", console.text)
        self.assertIn("Copy audio or video files", console.text)
        self.assertEqual(console.prompts, [])

    def test_failed_run_exits_one(self):
        self.media("a.wav")
        console = Console(["it", "", "", "y", ""])

        def make(config):
            self.configs.append(config)
            return TranscriptionPipeline(
                config,
                FakeProcessor(2),
                FakeBackend(failures={0, 1}),
                FileExporter(),
                status=console.say,
            )

        wizard = Wizard(
            self.config(),
            make,
            ask=console.ask,
            say=console.say,
            last_answers_path=self.last,
        )
        self.assertEqual(wizard.run(), 1)
        self.assertIn("status        : failed", console.text)


class LastAnswersTests(WizardCase):
    def test_defaults_are_loaded_and_answers_saved(self):
        self.media("a.wav")
        self.last.parent.mkdir(parents=True)
        self.last.write_text(
            json.dumps({"language": "en", "context": "old terms", "max_minutes": 5}),
            encoding="utf-8",
        )
        console = Console(["", "=", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        (config,) = self.run_configs()
        self.assertEqual(config.language, "en")
        self.assertEqual(config.prompt, "old terms")
        self.assertEqual(config.max_duration, 300.0)
        self.assertIn("[en]", console.prompts[0])
        self.assertIn("[5]", console.prompts[2])
        saved = load_last_answers(self.last)
        self.assertEqual(saved, {"language": "en", "context": "old terms", "max_minutes": 5.0})

    def test_unreadable_last_answers_are_ignored(self):
        self.media("a.wav")
        self.last.parent.mkdir(parents=True)
        self.last.write_text("{not json", encoding="utf-8")
        console = Console(["it", "", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertEqual(load_last_answers(self.last)["language"], "it")

    def test_config_language_is_the_default_without_history(self):
        self.media("a.wav")
        console = Console(["", "", "", "y", ""])
        self.wizard(console, self.config(language="it")).run()
        self.assertIn("[it]", console.prompts[0])
        self.assertEqual(self.run_configs()[0].language, "it")


class SummaryTests(WizardCase):
    def test_summary_table_has_statistics_and_no_transcript_text(self):
        self.media("a.wav")
        console = Console(["it", "", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        results = console.text.split("Results", 1)[1]
        for label in ("status", "version", "chunks", "confidence", "total time", "transcript"):
            self.assertIn(label, results)
        self.assertIn("2 ok, 0 failed", results)
        self.assertIn("uncalibrated", results)
        self.assertIn("run-report.md", results)
        self.assertNotIn("chunk 0", results)

    def test_queue_lists_size_duration_and_rough_estimate(self):
        self.media("a.wav")
        console = Console([])
        wizard = self.wizard(console)
        wizard._show_queue(wizard._queue())
        self.assertIn("a.wav", console.text)
        self.assertIn("rough estimate", console.text)
        self.assertIn("0.04", console.text)

    def test_estimate_uses_latest_completed_real_time_factor(self):
        self.media("a.wav")
        folder = self.root / "output" / "src-aaaaaaaaaaaa"
        folder.mkdir(parents=True)
        (self.root / "output" / "catalog.json").write_text(
            json.dumps({"sources": [{"folder": folder.name}]}), encoding="utf-8"
        )
        (folder / "source.json").write_text(
            json.dumps({"versions": [{"status": "completed", "real_time_factor": 0.5}]}),
            encoding="utf-8",
        )
        console = Console([])
        wizard = self.wizard(console)
        wizard._show_queue(wizard._queue())
        self.assertIn("from the most recent completed run", console.text)
        self.assertNotIn("rough estimate", console.text)

    def test_open_folder_is_asked_and_guarded(self):
        self.media("a.wav")
        opened = []
        console = Console(["it", "", "", "y", "y"])
        self.wizard(console, open_folder=opened.append).run()
        self.assertEqual(opened, [self.root / "output"])


class CliWizardTests(WizardCase):
    def args(self, *extra):
        return [
            "wizard",
            "--backend",
            "mock",
            "--input-dir",
            str(self.root / "input"),
            "--process-dir",
            str(self.root / "process"),
            "--processed-dir",
            str(self.root / "processed"),
            "--output-dir",
            str(self.root / "output"),
            *extra,
        ]

    def test_non_interactive_stdin_is_refused(self):
        stderr = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO("")),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(self.args())
        self.assertEqual(code, 2)
        self.assertIn("interactive console", stderr.getvalue())
        self.assertIn("run", stderr.getvalue())

    def test_answers_file_runs_without_a_console(self):
        self.media("a.wav")
        answers = self.root / "answers.json"
        answers.write_text(
            json.dumps({"language": "it", "max_minutes": 2, "confirm": True}), encoding="utf-8"
        )
        sink = Console([])
        stdout = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO("")),
            patch("audio_transcript.cli._make_pipeline", side_effect=self.factory(sink)),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(self.args("--answers", str(answers)))
        self.assertEqual(code, 0)
        self.assertEqual(self.run_configs()[0].language, "it")
        self.assertEqual(self.run_configs()[0].max_duration, 120.0)
        self.assertIn("completed", stdout.getvalue())

    def test_answers_file_with_confirm_false_cancels(self):
        self.media("a.wav")
        answers = self.root / "answers.json"
        answers.write_text(json.dumps({"confirm": False}), encoding="utf-8")
        sink = Console([])
        with (
            patch("audio_transcript.cli._make_pipeline", side_effect=self.factory(sink)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = main(self.args("--answers", str(answers)))
        self.assertEqual(code, 0)
        self.assertEqual(self.run_configs(), [])

    def test_invalid_answers_file_is_a_usage_error(self):
        answers = self.root / "answers.json"
        answers.write_text("[1]", encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main(self.args("--answers", str(answers))), 2)

    def test_language_auto_on_the_command_line(self):
        config = self.root / "c.toml"
        config.write_text('language = "it"\n', encoding="utf-8")
        loaded = load_config(config, {"language": "auto"})
        self.assertIsNone(loaded.language)


if __name__ == "__main__":
    unittest.main()
