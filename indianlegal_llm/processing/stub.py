"""Stub processor: a paragraph-aware chunker for Indian judgments.

It detects the numbered-paragraph structure of a judgment and chunks on
paragraph boundaries, recording each chunk's paragraph span in metadata
(``para_start``, ``para_end``) so answers can be pinpoint-cited (e.g.
"Puttaswamy para 297"). Paragraphs are grouped up to ``chunk_words``; a chunk
never breaks mid-paragraph unless a single paragraph exceeds the budget, in which
case it is windowed and every window keeps that paragraph's number.

Trivially replaceable with a richer layout-aware chunker behind
:class:`BaseProcessor` — the para_start/para_end contract is what matters.
"""

from __future__ import annotations

from ..schemas import Chunk, RawDoc
from .base import BaseProcessor
from ._paragraphs import Paragraph, segment_paragraphs


class StubProcessor(BaseProcessor):
    """Paragraph-boundary chunker.

    Parameters
    ----------
    chunk_words:
        Target maximum number of words per chunk. Paragraph boundaries are
        preferred over hitting this exactly.
    """

    def __init__(self, chunk_words: int = 90) -> None:
        if chunk_words <= 0:
            raise ValueError("chunk_words must be positive")
        self.chunk_words = chunk_words

    def _make_chunk(
        self,
        doc: RawDoc,
        index: int,
        text: str,
        para_start: int | None,
        para_end: int | None,
    ) -> Chunk:
        metadata = dict(doc.metadata)
        metadata["para_start"] = para_start
        metadata["para_end"] = para_end
        return Chunk(
            chunk_id=f"{doc.doc_id}::{index}",
            doc_id=doc.doc_id,
            text=text,
            title=doc.title,
            court=doc.court,
            url=doc.url,
            license=doc.license,
            metadata=metadata,
        )

    def process(self, doc: RawDoc) -> list[Chunk]:
        paragraphs = segment_paragraphs(doc.text)
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        index = 0
        buffer: list[Paragraph] = []
        buffer_words = 0

        def flush() -> None:
            nonlocal index, buffer, buffer_words
            if not buffer:
                return
            text = " ".join(p.text for p in buffer if p.text)
            numbers = [p.number for p in buffer if p.number is not None]
            para_start = numbers[0] if numbers else None
            para_end = numbers[-1] if numbers else None
            chunks.append(self._make_chunk(doc, index, text, para_start, para_end))
            index += 1
            buffer = []
            buffer_words = 0

        for para in paragraphs:
            words = para.text.split()
            # A single oversized paragraph: window it, keep its number on each part.
            if len(words) > self.chunk_words:
                flush()
                for start in range(0, len(words), self.chunk_words):
                    window = " ".join(words[start : start + self.chunk_words])
                    chunks.append(
                        self._make_chunk(doc, index, window, para.number, para.number)
                    )
                    index += 1
                continue
            # Otherwise pack paragraphs up to the budget, breaking on boundaries.
            if buffer and buffer_words + len(words) > self.chunk_words:
                flush()
            buffer.append(para)
            buffer_words += len(words)

        flush()
        return chunks
