"""Stub processor: a naive fixed-size word chunker with overlap.

Good enough to make retrieval meaningful in the skeleton; trivially replaceable
with a sentence/layout-aware chunker behind :class:`BaseProcessor`.
"""

from __future__ import annotations

from ..schemas import Chunk, RawDoc
from .base import BaseProcessor


class StubProcessor(BaseProcessor):
    """Split text into overlapping windows of whitespace-delimited tokens.

    Parameters
    ----------
    chunk_words:
        Target number of tokens per chunk.
    overlap_words:
        Number of tokens shared between consecutive chunks (keeps a sentence that
        straddles a boundary retrievable from either side).
    """

    def __init__(self, chunk_words: int = 60, overlap_words: int = 15) -> None:
        if chunk_words <= 0:
            raise ValueError("chunk_words must be positive")
        if not 0 <= overlap_words < chunk_words:
            raise ValueError("overlap_words must be in [0, chunk_words)")
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words

    def process(self, doc: RawDoc) -> list[Chunk]:
        tokens = doc.text.split()
        if not tokens:
            return []

        step = self.chunk_words - self.overlap_words
        chunks: list[Chunk] = []
        index = 0
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_words]
            if not window:
                break
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{index}",
                    doc_id=doc.doc_id,
                    text=" ".join(window),
                    title=doc.title,
                    court=doc.court,
                    url=doc.url,
                    license=doc.license,
                    metadata=dict(doc.metadata),
                )
            )
            index += 1
            # Stop once the window has consumed the tail of the document.
            if start + self.chunk_words >= len(tokens):
                break
        return chunks
