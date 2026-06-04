"""Small stdlib helpers shared by the real ingestors.

- ``html_to_text``: strip HTML to readable text without a third-party parser.
- ``safe_id``: normalize a raw source identifier (case number / CNR) into the
  citation charset the retriever validates (``[A-Za-z0-9_.:\\-]``), so a real
  document id can never produce an un-citable or invalid chunk_id.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def html_to_text(raw: str) -> str:
    """Return readable text from an HTML (or plain) string. Pure stdlib."""
    if not raw:
        return ""
    if "<" not in raw and "&" not in raw:
        return re.sub(r"\s+", " ", raw).strip()
    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except Exception:  # malformed HTML — fall back to unescaped, tag-stripped text
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()
    return parser.text()


_DMY_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")


def normalize_date(value: str) -> str:
    """Normalize a source date to ISO-8601 (YYYY-MM-DD) when recognizable.

    The AWS datasets store decision dates as ``DD-MM-YYYY`` (sometimes with
    unpadded day/month); convert those to zero-padded ISO for manifest
    consistency. Anything else (already-ISO, blank, unparseable) is returned
    trimmed and unchanged — we never guess an ambiguous date.
    """
    if not value:
        return ""
    text = str(value).strip()
    match = _DMY_RE.match(text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return text


_ID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.:\-]+")


def safe_id(raw: str) -> str:
    """Normalize a raw id into the citation charset; collapse runs to '-'.

    Returns an empty string if nothing usable remains (caller should skip the row).
    """
    if not raw:
        return ""
    cleaned = _ID_SANITIZE_RE.sub("-", raw.strip()).strip("-")
    return cleaned
