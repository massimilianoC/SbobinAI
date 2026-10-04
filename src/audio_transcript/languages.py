"""Language names typed by people, mapped to ISO codes.

Users answer "which language?" with a code (``it``), an English name (``Italian``),
the language's own name (``italiano``), a name in another guide language
(``inglese``, ``anglais``, ``Englisch``) or a region tag (``en-US``). This table covers
the languages of the Qwen3-ASR model; matching ignores case, accents and surrounding
spaces. It also lists the languages the guided wizard is translated into.
"""

from __future__ import annotations

import unicodedata

# Languages of the wizard's questions and messages (one JSON file each in
# locales/wizard/); "auto" follows the operating system.
UI_LANGUAGES = ("en", "it", "es", "fr", "de", "pt")
UI_LANGUAGE_CHOICES = ("auto", *UI_LANGUAGES)

# code: (English name, own name, other accepted names...)
_NAMES: dict[str, tuple[str, ...]] = {
    "ar": ("Arabic", "العربية", "arabo", "árabe", "arabe", "Arabisch"),
    "cs": ("Czech", "Čeština", "ceco", "checo", "tchèque", "Tschechisch", "tcheco"),
    "da": ("Danish", "Dansk", "danese", "danés", "danois", "Dänisch", "dinamarquês"),
    "de": ("German", "Deutsch", "tedesco", "alemán", "allemand", "alemão"),
    "el": ("Greek", "Ελληνικά", "greco", "griego", "grec", "Griechisch", "grego"),
    "en": ("English", "English", "inglese", "inglés", "anglais", "Englisch", "inglês"),
    "es": ("Spanish", "Español", "castellano", "spagnolo", "espagnol", "Spanisch", "espanhol"),
    "fa": ("Persian", "فارسی", "Farsi", "persiano", "persa", "persan", "Persisch"),
    "fi": ("Finnish", "Suomi", "finlandese", "finlandés", "finnois", "Finnisch", "finlandês"),
    "fil": ("Filipino", "Filipino", "Tagalog", "filippino", "philippin", "Philippinisch"),
    "fr": ("French", "Français", "francese", "francés", "Französisch", "francês"),
    "hi": ("Hindi", "हिन्दी"),
    "hu": ("Hungarian", "Magyar", "ungherese", "húngaro", "hongrois", "Ungarisch"),
    "id": ("Indonesian", "Bahasa Indonesia", "indonesiano", "indonesio", "indonésien"),
    "it": ("Italian", "Italiano", "italien", "Italienisch"),
    "ja": ("Japanese", "日本語", "giapponese", "japonés", "japonais", "Japanisch", "japonês"),
    "ko": ("Korean", "한국어", "coreano", "coréen", "Koreanisch"),
    "mk": ("Macedonian", "Македонски", "macedone", "macedonio", "macédonien", "Mazedonisch"),
    "ms": ("Malay", "Bahasa Melayu", "malese", "malayo", "malais", "Malaiisch", "malaio"),
    "nl": ("Dutch", "Nederlands", "olandese", "neerlandés", "néerlandais", "Niederländisch"),
    "pl": ("Polish", "Polski", "polacco", "polaco", "polonais", "Polnisch"),
    "pt": ("Portuguese", "Português", "portoghese", "portugués", "portugais", "Portugiesisch"),
    "ro": ("Romanian", "Română", "rumeno", "rumano", "roumain", "Rumänisch", "romeno"),
    "ru": ("Russian", "Русский", "russo", "ruso", "russe", "Russisch"),
    "sv": ("Swedish", "Svenska", "svedese", "sueco", "suédois", "Schwedisch"),
    "th": ("Thai", "ไทย", "thailandese", "tailandés", "thaï", "Thailändisch", "tailandês"),
    "tr": ("Turkish", "Türkçe", "turco", "turc", "Türkisch"),
    "vi": ("Vietnamese", "Tiếng Việt", "vietnamita", "vietnamien", "Vietnamesisch"),
    "yue": ("Cantonese", "粵語", "廣東話", "cantonés", "cantonais", "Kantonesisch", "cantonês"),
    "zh": ("Chinese", "中文", "Mandarin", "cinese", "chino", "chinois", "Chinesisch", "chinês"),
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


def own_name(code: str | None) -> str | None:
    """The language's name in itself (``it`` -> ``Italiano``), or None when unknown."""
    names = _NAMES.get(code or "")
    return names[1] if names else None
