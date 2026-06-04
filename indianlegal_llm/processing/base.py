"""Processor interface.

A processor splits a :class:`RawDoc` into a list of :class:`Chunk` objects. Real
processors might do sentence-aware or layout-aware chunking, deduplication, or
paragraph detection — always behind this same interface, always preserving
provenance onto every chunk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Chunk, RawDoc


class BaseProcessor(ABC):
    """Abstract document -> chunks transformer."""

    @abstractmethod
    def process(self, doc: RawDoc) -> list[Chunk]:
        """Split a single raw document into retrievable chunks.

        Implementations MUST copy provenance (title, court, url, license) from the
        ``doc`` onto every chunk, and MUST produce ``chunk_id`` values that are
        unique within the corpus and free of square brackets.
        """
        raise NotImplementedError
