"""Translated texts of the guided wizard and the choice of their language.

One JSON file per language lives in ``audio_transcript/locales/wizard/<code>.json``;
``en.json`` is the reference. Missing keys fall back to English, so a partial
translation still works. To add a language, copy ``en.json``, translate the values
(keep every ``{placeholder}``), and add the code to ``UI_LANGUAGES`` in
``audio_transcript/languages.py`` (the CLI choices and the configuration check follow).

With ``ui_language = "auto"`` the language follows the operating system: the Windows
display language, otherwise ``LC_ALL``, ``LC_MESSAGES``, ``LANG`` or ``LANGUAGE``. An
unsupported system language falls back to English. Presentation only: the guide
language never changes the transcript or the job identity.
"""

from __future__ import annotations

import ctypes
import functools
import json
import locale
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from ..languages import UI_LANGUAGE_CHOICES, UI_LANGUAGES, language_code

__all__ = ["UI_LANGUAGE_CHOICES", "UI_LANGUAGES", "Texts", "resolve_ui_language"]
LOCALE_DIR = Path(__file__).resolve().parent.parent / "locales" / "wizard"

# Answers accepted in any guide language, so a reply never depends on the guide language.
YES_WORDS = frozenset({"y", "yes", "s", "si", "sì", "sí", "sim", "o", "oui", "j", "ja"})
FULL_WORDS = frozenset(
    {"full", "f", "all", "intero", "tutto", "completo", "todo", "entier", "tout", "ganz",
     "alles", "inteiro", "tudo"}
)  # fmt: skip


@functools.cache
def load_messages(language: str) -> dict[str, str]:
    """Messages of one language; an unknown or unreadable file gives an empty mapping."""
    try:
        data = json.loads((LOCALE_DIR / f"{language}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return {}
    return {key: value for key, value in data.items() if isinstance(value, str)}


class Texts:
    """``texts("key", name=value)`` returns the translated, formatted message."""

    def __init__(self, language: str = "en") -> None:
        self.language = language if language in UI_LANGUAGES else "en"
        self._english = load_messages("en")
        self._own = load_messages(self.language) if self.language != "en" else {}

    def __call__(self, key: str, **values: object) -> str:
        template = self._own.get(key) or self._english[key]
        return template.format(**values)

    @property
    def name(self) -> str:
        return self("_language")


def _windows_ui_locale() -> str | None:
    try:
        lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    return locale.windows_locale.get(lcid)


def system_ui_language(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    windows_locale: object = _windows_ui_locale,
) -> str:
    """Guide language from the operating system, or ``en`` when it is not translated."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    candidates: list[str | None] = []
    if platform == "win32":
        candidates.append(windows_locale() if callable(windows_locale) else None)
    for variable in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = environ.get(variable)
        if value:
            candidates.append(value.split(":")[0].split(".")[0])
    if platform != "win32":
        candidates.append(locale.getlocale()[0])
    for candidate in candidates:
        code = language_code(candidate)
        if code in UI_LANGUAGES:
            return code
    return "en"


def resolve_ui_language(setting: str | None) -> tuple[str, bool]:
    """Return ``(language, detected)`` for the ``ui_language`` setting."""
    if not setting or setting == "auto":
        return system_ui_language(), True
    code = language_code(setting)
    return (code if code in UI_LANGUAGES else "en"), False
