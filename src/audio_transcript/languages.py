"""Language names typed by people, mapped to ISO codes.

Users answer "which language?" with a code (``it``), an English name (``Italian``),
the language's own name (``italiano``) or, in an Italian console, an Italian name
(``inglese``). This table covers the languages of the Qwen3-ASR model; matching
ignores case, accents and surrounding spaces, and accepts region tags (``it-IT``).
"""

from __future__ import annotations

import unicodedata

# code: (English name, own name(s), Italian name)
_NAMES: dict[str, tuple[str, ...]] = {
    "ar": ("Arabic", "العربية", "arabo"),
    "cs": ("Czech", "čeština", "ceco"),
    "da": ("Danish", "dansk", "danese"),
    "de": ("German", "Deutsch", "tedesco"),
    "el": ("Greek", "ελληνικά", "greco"),
    "en": ("English", "inglese"),
    "es": ("Spanish", "español", "castellano", "spagnolo"),
    "fa": ("Persian", "فارسی", "Farsi", "persiano"),
    "fi": ("Finnish", "suomi", "finlandese"),
    "fil": ("Filipino", "Tagalog", "filippino"),
    "fr": ("French", "français", "francese"),
    "hi": ("Hindi", "हिन्दी"),
    "hu": ("Hungarian", "magyar", "ungherese"),
    "id": ("Indonesian", "Bahasa Indonesia", "indonesiano"),
    "it": ("Italian", "italiano"),
    "ja": ("Japanese", "日本語", "giapponese"),
    "ko": ("Korean", "한국어", "coreano"),
    "mk": ("Macedonian", "македонски", "macedone"),
    "ms": ("Malay", "Bahasa Melayu", "malese"),
    "nl": ("Dutch", "Nederlands", "olandese"),
    "pl": ("Polish", "polski", "polacco"),
    "pt": ("Portuguese", "português", "portoghese"),
    "ro": ("Romanian", "română", "rumeno"),
    "ru": ("Russian", "русский", "russo"),
    "sv": ("Swedish", "svenska", "svedese"),
    "th": ("Thai", "ไทย", "thailandese"),
    "tr": ("Turkish", "Türkçe", "turco"),
    "vi": ("Vietnamese", "Tiếng Việt", "vietnamita"),
    "yue": ("Cantonese", "粵語", "廣東話"),
    "zh": ("Chinese", "中文", "Mandarin", "cinese"),
}


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.strip().casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


_BY_NAME: dict[str, str] = {}
for _code, _names in _NAMES.items():
    _BY_NAME[_fold(_code)] = _code
    for _name in _names:
        _BY_NAME.setdefault(_fold(_name), _code)


def language_code(text: str | None) -> str | None:
    """Return the ISO code for a typed code or name, or None when it is not known."""
    if not text:
        return None
    key = _fold(text)
    if key in _BY_NAME:
        return _BY_NAME[key]
    for separator in ("-", "_"):
        base = key.split(separator, 1)[0]
        if base != key and base in _BY_NAME:
            return _BY_NAME[base]
    return None
