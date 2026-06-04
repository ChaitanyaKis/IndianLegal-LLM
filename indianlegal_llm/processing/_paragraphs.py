"""Detect numbered-paragraph structure in Indian judgments.

Indian judgments are paragraph-numbered, and those numbers are how lawyers
pinpoint-cite (e.g. "Puttaswamy para 297"). This module segments judgment text
into ``Paragraph(number, text)`` units so the chunker can keep each chunk's
paragraph span (para_start, para_end) and prefer paragraph boundaries.

Markers detected at a line start (optionally after a pilcrow), or a pilcrow
followed by a number anywhere::

    1.  text...      2) text...      ¶ 297 text...      297. text...

Text before the first marker is an unnumbered preamble (number=None). Detection
works best on text that retains line breaks; ``ingestion._util.html_to_text``
preserves block-level breaks precisely so this can find structure in real
``raw_html``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Branch A: number + '.'/')' at a line start (optionally prefixed by a pilcrow).
# Branch B: a pilcrow immediately followed by a number, anywhere.
_PARA_RE = re.compile(
    r"(?:(?:\A|\n)[ \t]*(?:¶[ \t]*)?(?P<a>\d{1,4})[.)][ \t]+)"
    r"|(?:¶[ \t]*(?P<b>\d{1,4})\b[ \t]*)"
)


@dataclass
class Paragraph:
    """A judgment paragraph; ``number`` is None for unnumbered preamble text."""

    number: int | None
    text: str


def _normalize(text: str) -> str:
    return " ".join(text.split())


def segment_paragraphs(text: str) -> list[Paragraph]:
    """Split judgment text into numbered paragraphs (preamble first, if any)."""
    if not text or not text.strip():
        return []

    matches = list(_PARA_RE.finditer(text))
    if not matches:
        return [Paragraph(None, _normalize(text))]

    paragraphs: list[Paragraph] = []
    preamble = _normalize(text[: matches[0].start()])
    if preamble:
        paragraphs.append(Paragraph(None, preamble))

    for i, match in enumerate(matches):
        number = int(match.group("a") or match.group("b"))
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _normalize(text[body_start:body_end])
        if body:
            paragraphs.append(Paragraph(number, body))
    return paragraphs
