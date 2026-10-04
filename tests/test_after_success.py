"""after_success: what happens to a queued source after a complete transcription."""

import json
import tempfile
import unittest
from pathlib import Path

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.application.wizard import parse_after_success
from audio_transcript.config import load_config

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
    from .test_wizard import Console, WizardCase
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeBackend, FakeProcessor, make_config
    from test_wizard import Console, WizardCase


class AfterSuccessPipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.source = self.root / "input" / "talk.mp4"
        self.source.parent.mkdir()
        self.source.write_bytes(b"original media bytes")

    def run_job(self, processor=None, source=None, **overrides):
        backend = FakeBackend()
        backend.name = "llamacpp"
        config = make_config(self.root, backend="llamacpp", model="qwen3-asr", **overrides)
        processor = processor or FakeProcessor(chunk_count=2)
        pipeline = TranscriptionPipeline(
            config, processor, backend, FileExporter(), status=lambda *_: None
        )
        result = pipeline.run([source or self.source])[0]
        self.assertEqual(result["status"], "completed")
        return result

    def folders(self, result):
        source_folder = result["job_id"].split("/")[0]
        return (
            self.root / "processed" / source_folder,
            self.root / "process" / result["job_id"],
        )

    def metadata(self, result):
        return json.loads((self.folders(result)[1] / "metadata.json").read_text(encoding="utf-8"))

    def test_keep_all_is_the_default_and_moves_the_original(self):
        result = self.run_job()
        processed, work = self.folders(result)
        self.assertTrue((processed / "talk.mp4").is_file())
        self.assertTrue(any(work.glob("chunk_*.wav")))
        self.assertNotIn("source_disposition", result)
        self.assertIn("archived_source_path", result)

    def test_keep_audio_keeps_a_verified_flac_and_deletes_the_original(self):
        processor = FakeProcessor(chunk_count=2)
        result = self.run_job(processor, after_success="keep-audio")
        processed, work = self.folders(result)
        self.assertFalse((processed / "talk.mp4").exists())
        self.assertTrue((processed / "talk.audio.flac").is_file())
        self.assertEqual(processor.export_calls[0][2], 16000)
        self.assertFalse(any(work.glob("chunk_*.wav")))
        disposition = self.metadata(result)["source_disposition"]
        self.assertEqual(disposition["policy"], "keep-audio")
        self.assertTrue(disposition["original_deleted"])
        self.assertTrue(disposition["audio_path"].endswith("talk.audio.flac"))
        self.assertNotIn("archived_source_path", self.metadata(result))
        self.assertTrue((self.root / "output" / result["job_id"] / "transcript.txt").is_file())

    def test_delete_all_deletes_original_and_chunk_audio_but_keeps_results(self):
        result = self.run_job(after_success="delete-all")
        processed, work = self.folders(result)
        self.assertEqual(list(processed.iterdir()), [])
        self.assertFalse(any(work.glob("chunk_*.wav")))
        self.assertTrue((work / "checkpoint.json").is_file())
        self.assertEqual(result["source_disposition"]["policy"], "delete-all")
        self.assertTrue((self.root / "output" / result["job_id"] / "report.json").is_file())

    def test_failed_or_short_audio_export_keeps_the_original(self):
        for label, processor in (
            ("error", FakeProcessor(chunk_count=2)),
            ("short", FakeProcessor(chunk_count=2)),
        ):
            with self.subTest(label):
                if label == "error":
                    processor.export_error = "disk full"
                else:
                    processor.export_duration = 0.5
                self.source.write_bytes(f"original {label}".encode())
                result = self.run_job(processor, after_success="keep-audio")
                processed, work = self.folders(result)
                self.assertTrue((processed / "talk.mp4").is_file())
                self.assertFalse((processed / "talk.audio.flac").exists())
                self.assertTrue(any(work.glob("chunk_*.wav")))
                self.assertIn("the original was kept", result["disposition_warning"])
                self.assertIn("archived_source_path", result)

    def test_bounded_runs_and_files_outside_input_are_never_deleted(self):
        # Even a bounded run that happens to cover the whole file stays in input.
        result = self.run_job(
            FakeProcessor(chunk_count=1), after_success="delete-all", max_duration=5.0
        )
        self.assertTrue(self.source.is_file())
        self.assertNotIn("source_disposition", result)
        self.assertNotIn("archived_source_path", result)
        self.assertIn("Bounded run", result["archive_warning"])
        outside = self.root / "elsewhere.mp4"
        outside.write_bytes(b"outside media")
        self.run_job(source=outside, after_success="delete-all")
        self.assertTrue(outside.is_file())

    def test_config_validation(self):
        self.assertEqual(
            load_config(None, {"after_success": "keep-audio"}).after_success, "keep-audio"
        )
        with self.assertRaises(ValueError):
            load_config(None, {"after_success": "remove"})


class AfterSuccessWizardTests(WizardCase):
    def config(self, **overrides):
        return super().config(**overrides)

    def test_parse_numbers_and_words(self):
        self.assertEqual(parse_after_success("1"), "keep-all")
        self.assertEqual(parse_after_success(" Audio "), "keep-audio")
        self.assertEqual(parse_after_success("elimina"), "delete-all")
        self.assertEqual(parse_after_success("löschen"), "delete-all")
        self.assertIsNone(parse_after_success("maybe"))

    def test_choice_is_asked_for_full_files_and_warned(self):
        self.media("a.wav")
        console = Console(["it", "", "", "x", "2", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        (config,) = self.run_configs()
        self.assertEqual(config.after_success, "keep-audio")
        self.assertIn("Type 1, 2 or 3", console.text)
        self.assertIn("deleted permanently", console.text)
        self.assertIn("afterwards      : delete the original, keep only its audio", console.text)
        self.assertIn("(default)", console.text)

    def test_not_asked_for_bounded_scope(self):
        self.media("a.wav")
        console = Console(["it", "", "5", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertFalse(any("Original file" in prompt for prompt in console.prompts))
        self.assertEqual(self.run_configs()[0].after_success, "keep-all")

    def test_not_asked_for_files_outside_the_input_folder(self):
        outside = self.root / "elsewhere.wav"
        outside.write_bytes(b"synthetic outside")
        console = Console(["it", "", "", "y", ""])
        self.assertEqual(self.wizard(console, sources=[outside]).run(), 0)
        self.assertFalse(any("Original file" in prompt for prompt in console.prompts))

    def test_default_comes_from_configuration_and_is_not_remembered(self):
        self.media("a.wav")
        console = Console(["it", "", "", "", "y", ""])
        self.wizard(console, self.config(after_success="delete-all")).run()
        self.assertIn("[3]", console.prompts[3])
        self.assertEqual(self.run_configs()[0].after_success, "delete-all")
        saved = json.loads(self.last.read_text(encoding="utf-8"))
        self.assertNotIn("after_success", saved)

    def test_scripted_answers(self):
        self.media("a.wav")
        console = Console([])
        wizard = self.wizard(console, scripted={"language": "it", "after_success": "audio"})
        self.assertEqual(wizard.run(), 0)
        self.assertEqual(self.run_configs()[0].after_success, "keep-audio")
        console = Console([])
        wizard = self.wizard(console, scripted={"after_success": "later"})
        self.assertEqual(wizard.run(), 2)


class LanguageMenuTests(WizardCase):
    def test_numbers_pick_from_the_menu_and_bad_numbers_are_explained(self):
        self.media("a.wav")
        console = Console(["9", "1", "", "", "", "y", ""])
        self.assertEqual(self.wizard(console).run(), 0)
        self.assertIsNone(self.run_configs()[0].language)  # 1 = auto
        self.assertIn("There is no option 9", console.text)
        self.assertIn("   2  en    English", console.text)
        self.assertIn("   3  it    Italiano", console.text)
        self.assertIn("Other languages, by code:", console.text)
        self.assertIn("Examples:", console.text)

    def test_guide_language_is_listed_first(self):
        from audio_transcript.application.wizard_text import Texts

        self.media("a.wav")
        console = Console(["2", "", "", "", "s", ""])
        self.assertEqual(self.wizard(console, texts=Texts("it")).run(), 0)
        self.assertEqual(self.run_configs()[0].language, "it")
        self.assertIn("   2  it    Italiano", console.text)


if __name__ == "__main__":
    unittest.main()
