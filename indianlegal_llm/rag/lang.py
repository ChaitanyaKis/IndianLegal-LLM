"""Indian-language detection for the cross-lingual answering path.

Detects the script of a question (Hindi/Telugu/Tamil/Malayalam/...) so the
Answerer can instruct the model to explain in the user's language WHILE keeping
quoted authority verbatim in its source language (English). Pure stdlib (Unicode
block ranges) — deterministic, no external language-detection dependency.
"""

from __future__ import annotations

# Major Indian-script Unicode blocks -> ISO-639-1 code. Devanagari covers Hindi
# (and Marathi/etc.); we label it "hi" for the explanation-language directive.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("hi", 0x0900, 0x097F),  # Devanagari
    ("bn", 0x0980, 0x09FF),  # Bengali
    ("pa", 0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    ("gu", 0x0A80, 0x0AFF),  # Gujarati
    ("or", 0x0B00, 0x0B7F),  # Odia
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("te", 0x0C00, 0x0C7F),  # Telugu
    ("kn", 0x0C80, 0x0CFF),  # Kannada
    ("ml", 0x0D00, 0x0D7F),  # Malayalam
)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
}


def detect_language(text: str) -> str:
    """Return the dominant Indian-script language code, or "en" if none/Latin."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return "en"
    return max(counts, key=counts.get)


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)
