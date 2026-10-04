"""Typed language codes and names mapped to ISO codes."""

import unittest

from audio_transcript.languages import language_code


class LanguageCodeTests(unittest.TestCase):
    def test_codes_english_own_and_italian_names(self):
        cases = {
            "it": "it",
            "Italian": "it",
            "italiano": "it",
            "inglese": "en",
            "Deutsch": "de",
            "tedesco": "de",
            "français": "fr",
            "francais": "fr",
            "ESPAÑOL": "es",
            "русский": "ru",
            "中文": "zh",
        }
        for typed, code in cases.items():
            with self.subTest(typed=typed):
                self.assertEqual(language_code(typed), code)

    def test_region_tags_and_spaces(self):
        self.assertEqual(language_code(" it-IT "), "it")
        self.assertEqual(language_code("pt_BR"), "pt")

    def test_unknown_or_empty(self):
        self.assertIsNone(language_code("klingon"))
        self.assertIsNone(language_code(""))
        self.assertIsNone(language_code(None))


if __name__ == "__main__":
    unittest.main()
