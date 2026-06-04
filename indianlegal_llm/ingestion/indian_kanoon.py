"""Indian Kanoon ingestor (judgments / bare acts — government works).

Fetches public document pages at ``https://indiankanoon.org/doc/<id>/`` and maps
them to :class:`RawDoc`. Indian Kanoon reproduces government works, which are
license-clean per CLAUDE.md §3. Forbidden sources (SCC Online, Manupatra, ILDC)
are never touched.

The network fetch imports its HTTP client lazily (``[ingestion]`` extra). The
pure ``doc_to_rawdoc`` mapper is what the unit tests exercise against a committed
HTML fixture — no network needed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..schemas import RawDoc
from ._http import http_get
from ._util import html_to_text, safe_id
from .base import BaseIngestor

_LICENSE = "government-work"
_DOC_URL = "https://indiankanoon.org/doc/{doc_id}/"

# Real, verifiable seed doc ids (the two landmark judgments also used by the
# stub). Override via the constructor / CLI for a real crawl.
_DEFAULT_DOC_IDS = ("91938676", "257876")


class IndianKanoonIngestor(BaseIngestor):
    """Fetch a configurable list of Indian Kanoon document ids."""

    source_name = "indian-kanoon"

    def __init__(self, limit: int = 200, *, doc_ids: Sequence[str] | None = None) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.doc_ids = list(doc_ids) if doc_ids is not None else list(_DEFAULT_DOC_IDS)

    def fetch(self) -> Iterator[RawDoc]:
        for doc_id in self.doc_ids[: self.limit]:
            html = http_get(_DOC_URL.format(doc_id=doc_id))
            doc = self.doc_to_rawdoc(str(doc_id), html)
            if doc is not None:
                yield doc

    @staticmethod
    def doc_to_rawdoc(doc_id: str, html: str) -> RawDoc | None:
        """Map an Indian Kanoon document page to a RawDoc (pure; testable)."""
        ident = safe_id(doc_id)
        if not ident:
            return None
        text = html_to_text(html)
        if not text:
            return None
        # The page <title> is "<Case Title> - Indian Kanoon"; take the lead.
        title = text.split(" - Indian Kanoon")[0][:200].strip() or f"doc/{doc_id}"
        lower = text.lower()
        if "supreme court" in lower:
            court = "Supreme Court of India"
        elif "high court" in lower:
            court = "High Court (India)"
        else:
            court = "Indian Kanoon (court unspecified)"
        return RawDoc(
            doc_id=f"ik-{ident}",
            title=title,
            court=court,
            date="",  # decision date parsing is source-specific; left blank here
            url=_DOC_URL.format(doc_id=doc_id),
            license=_LICENSE,
            text=text,
            language="en",
            metadata={"source": "indian-kanoon", "ik_doc_id": str(doc_id)},
        )
