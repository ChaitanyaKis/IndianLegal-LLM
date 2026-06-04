"""India Code ingestor (central/state bare acts — government works).

India Code (https://www.indiacode.nic.in) is the official repository of Indian
statutes; bare acts are government works and license-clean per CLAUDE.md §3.

This fetches a configurable list of act page URLs and maps each to a
:class:`RawDoc`. Because India Code's exact per-act handle ids must be looked up
(do not guess them), the default seed list is EMPTY: pass act URLs via the
constructor or the ingestion CLI's ``--urls``. The pure ``page_to_rawdoc`` mapper
is unit-tested against a committed HTML fixture; the network fetch imports its
HTTP client lazily (``[ingestion]`` extra).

Forbidden sources (SCC Online, Manupatra, ILDC) are never touched.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from urllib.parse import urlparse

from ..schemas import RawDoc
from ._http import http_get
from ._util import html_to_text, safe_id
from .base import BaseIngestor

_LICENSE = "government-work"
_COURT = "India Code (bare act)"
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
# Jurisdiction guard (CLAUDE.md §1/§3): only the official India Code host may be
# stamped as an India Code government work. Mirrors the hard-bound source of the
# AWS (fixed buckets) and Indian Kanoon (fixed URL template) ingestors.
_ALLOWED_HOST = "indiacode.nic.in"


def _is_india_code_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == _ALLOWED_HOST or host.endswith("." + _ALLOWED_HOST)


class IndiaCodeIngestor(BaseIngestor):
    """Fetch a configurable list of India Code act page URLs."""

    source_name = "india-code"

    def __init__(self, limit: int = 200, *, urls: Sequence[str] | None = None) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.urls = list(urls) if urls is not None else []

    def fetch(self) -> Iterator[RawDoc]:
        if not self.urls:
            raise ValueError(
                "IndiaCodeIngestor needs act URLs. Pass urls=[...] or the CLI "
                "--urls flag (India Code handle ids must be looked up, not guessed)."
            )
        for url in self.urls[: self.limit]:
            if not _is_india_code_url(url):
                raise ValueError(
                    f"refusing non-India-Code URL {url!r}: only {_ALLOWED_HOST} "
                    "is allowed (jurisdiction/provenance guard, CLAUDE.md §1/§3)."
                )
            html = http_get(url)
            doc = self.page_to_rawdoc(url, html)
            if doc is not None:
                yield doc

    @staticmethod
    def page_to_rawdoc(url: str, html: str) -> RawDoc | None:
        """Map an India Code act page to a RawDoc (pure; testable)."""
        ident = safe_id(url.rstrip("/").split("/")[-1])
        if not ident:
            return None
        text = html_to_text(html)
        if not text:
            return None
        title = text.split(" | ")[0][:200].strip() or f"act/{ident}"
        year_match = _YEAR_RE.search(title) or _YEAR_RE.search(text[:400])
        date = year_match.group(0) if year_match else ""
        return RawDoc(
            doc_id=f"ic-{ident}",
            title=title,
            court=_COURT,
            date=date,
            url=url,
            license=_LICENSE,
            text=text,
            language="en",
            metadata={"source": "india-code", "act_url": url},
        )
