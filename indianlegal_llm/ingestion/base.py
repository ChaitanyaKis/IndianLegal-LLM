"""Ingestor interface.

An ingestor yields :class:`RawDoc` objects from a license-clean source. Real
ingestors stream from AWS Open Data (Indian SC/HC judgments), India Code, or
Indian Kanoon (CLAUDE.md §3). They MUST set accurate provenance fields
(``url``, ``court``, ``date``, ``license``) on every document so the ingestion
manifest is complete.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..schemas import RawDoc


class BaseIngestor(ABC):
    """Abstract source of raw Indian legal documents."""

    #: Human-readable name of the source, for the manifest/logs.
    source_name: str = "base"

    @abstractmethod
    def fetch(self) -> Iterator[RawDoc]:
        """Yield raw documents one at a time.

        Implementations should stream (yield) rather than build a giant list, to
        honour the bandwidth constraint in CLAUDE.md §5 (corpora are processed in
        the cloud, never materialised wholesale on a laptop).
        """
        raise NotImplementedError

    def manifest(self) -> list[dict]:
        """Provenance manifest for everything this ingestor yields."""
        return [doc.manifest_entry() for doc in self.fetch()]
