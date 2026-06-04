"""InMemoryRetriever: a stdlib lexical retriever (stopword-filtered token overlap).

Scoring is the summed term-frequency overlap between the query's content tokens
and each chunk. Chunks with zero overlap are omitted entirely — that omission is
what lets the Answerer refuse out-of-domain questions (CLAUDE.md §4). Replaceable
by a vector retriever behind :class:`BaseRetriever`.
"""

from __future__ import annotations

import re
from collections import Counter

from ..schemas import Chunk, RetrievedChunk
from .base import BaseRetriever

# A small, general English stopword list. Kept deliberately conservative so that
# legally meaningful words (e.g. "right", "structure") are never filtered out.
STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how in into is it its of on or
    that the their this to was were what when where which who why will with
    do does did done can could should would may might must shall about over under
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A chunk_id must be citable: non-empty and confined to the citation charset
# (the same charset extract_cited_ids matches). This is validated on add() so a
# malformed id can never silently become an un-citable source (false refusal) or
# collide with another document (mis-attribution). See CLAUDE.md §4.
_VALID_CHUNK_ID = re.compile(r"^[A-Za-z0-9_.:\-]+$")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in STOPWORDS
    ]


class InMemoryRetriever(BaseRetriever):
    """Keeps chunks in a list and ranks them by lexical overlap with the query.

    A chunk is only returned if it shares at least ``min_overlap`` *distinct*
    content tokens with the query (or all of them, for very short queries). This
    relevance gate is what lets an out-of-domain question — which shares at most
    an incidental token or two with the Indian-law corpus — retrieve nothing and
    therefore be refused (CLAUDE.md §4).

    Note: lexical overlap is a coarse stub. It cannot tell "privacy under the US
    constitution" from "privacy under the Indian constitution" when they share
    several tokens; true jurisdiction-aware relevance arrives with the semantic
    retriever (see docs/ROADMAP.md, Milestone 3). The gate below removes the
    single-incidental-token failure class, not all of them.
    """

    def __init__(self, min_overlap: int = 2) -> None:
        if min_overlap < 1:
            raise ValueError("min_overlap must be >= 1")
        self.min_overlap = min_overlap
        self._chunks: list[Chunk] = []
        self._counts: list[Counter[str]] = []
        self._seen_ids: set[str] = set()

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            cid = chunk.chunk_id
            if not _VALID_CHUNK_ID.match(cid):
                raise ValueError(
                    f"invalid chunk_id {cid!r}: must be non-empty and match "
                    r"[A-Za-z0-9_.:\-]+ (no spaces, brackets, or other characters)"
                )
            if cid in self._seen_ids:
                raise ValueError(
                    f"duplicate chunk_id {cid!r}: chunk_ids must be unique across "
                    "the corpus (CLAUDE.md §4 — a citation must map to one source)"
                )
            self._seen_ids.add(cid)
            self._chunks.append(chunk)
            self._counts.append(Counter(tokenize(chunk.text)))

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_tokens = set(tokenize(query))
        if not query_tokens or top_k <= 0:
            return []

        # Short queries can't be asked to overlap on more tokens than they have.
        required = min(self.min_overlap, len(query_tokens))

        scored: list[RetrievedChunk] = []
        for chunk, counts in zip(self._chunks, self._counts):
            distinct_overlap = sum(1 for tok in query_tokens if counts[tok] > 0)
            if distinct_overlap < required:
                continue  # not relevant enough — omit (lets out-of-domain refuse)
            score = sum(counts[tok] for tok in query_tokens)
            scored.append(RetrievedChunk(chunk=chunk, score=float(score)))

        # Highest score first; ties keep insertion order (Python's sort is stable),
        # which means earlier-ingested documents win ties deterministically.
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:top_k]
