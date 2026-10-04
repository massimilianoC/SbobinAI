"""Wizard translations: catalogue consistency, system language detection, Italian guide."""

import json
import string
import unittest

from audio_transcript.application.wizard_text import (
    LOCALE_DIR,
    UI_LANGUAGES,
    Texts,
    resolve_ui_language,
    system_ui_language,
)
from audio_transcript.config import load_config

try:
    from .test_wizard import Console, WizardCase
except ImportError:  # discovered with ``-s tests``
    from test_wizard import Console, WizardCase


def _placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


class CatalogueTests(unittest.TestCase):
    def test_every_language_has_a_file_with_the_english_keys_and_placeholders(self):
        english = json.loads((LOCALE_DIR / "en.json").read_text(encoding="utf-8"))
        for language in UI_LANGUAGES:
            with self.subTest(language=language):
                data = json.loads((LOCALE_DIR / f"{language}.json").read_text(encoding="utf-8"))
                self.assertEqual(set(data), set(english))
                for key, text in english.items():
                    self.assertEqual(_placeholders(data[key]), _placeholders(text), key)

    def test_no_stray_locale_files(self):
        self.assertEqual({path.stem for path in LOCALE_DIR.glob("*.json")}, set(UI_LANGUAGES))

    def test_unknown_language_falls_back_to_english(self):
        texts = Texts("xx")
        self.assertEqual(texts.language, "en")
        self.assertEqual(texts("summary_header"), "Summary")
        self.assertEqual(Texts("it")("summary_header"), "Riepilogo")


class DetectionTests(unittest.TestCase):
    def test_windows_display_language_wins(self):
        language = system_ui_language(
            {"LANG": "en_US.UTF-8"}, "win32", windows_locale=lambda: "it_IT"
        )
        self.assertEqual(language, "it")

    def test_posix_variables(self):
        self.assertEqual(system_ui_language({"LANG": "de_DE.UTF-8"}, "linux"), "de")
        self.assertEqual(system_ui_language({"LC_ALL": "fr_FR.UTF-8", "LANG": "it"}, "linux"), "fr")
        self.assertEqual(system_ui_language({"LANGUAGE": "pt_BR:en"}, "darwin"), "pt")

    def test_untranslated_or_unknown_system_language_gives_english(self):
        self.assertEqual(system_ui_language({}, "win32", windows_locale=lambda: "ja_JP"), "en")
        self.assertEqual(system_ui_language({}, "win32", windows_locale=lambda: None), "en")

    def test_setting_overrides_detection(self):
        self.assertEqual(resolve_ui_language("es"), ("es", False))
        self.assertTrue(resolve_ui_language("auto")[1])

    def test_config_rejects_unknown_guide_language(self):
        self.assertEqual(load_config(None, {"ui_language": "it"}).ui_language, "it")
        with self.assertRaises(ValueError):
            load_config(None, {"ui_language": "xx"})


class ItalianGuideTests(WizardCase):
    def test_italian_questions_and_answers(self):
        self.media("a.wav")
        console = Console(["italiano", "Nomi: Rossi", "intero", "s", ""])
        wizard = self.wizard(console, texts=Texts("it"), language_detected=True)
        self.assertEqual(wizard.run(), 0)
        self.assertIn("Lingua parlata", console.prompts[0])
        self.assertIn("Avviare la trascrizione?", console.prompts[3])
        self.assertIn("Lingua della guida: Italiano (dalla lingua del sistema)", console.text)
        self.assertIn("Riepilogo", console.text)
        self.assertIn("Risultati", console.text)
        (config,) = self.run_configs()
        self.assertEqual(config.language, "it")
        self.assertEqual(config.prompt, "Nomi: Rossi")
        self.assertIsNone(config.max_duration)

    def test_italian_errors_are_translated(self):
        self.media("a.wav")
        console = Console(["klingon", "it", "", "0", "", "s", ""])
        self.assertEqual(self.wizard(console, texts=Texts("it")).run(), 0)
        self.assertIn("non è una lingua supportata", console.text)
        self.assertIn("maggiore di zero", console.text)


if __name__ == "__main__":
    unittest.main()
