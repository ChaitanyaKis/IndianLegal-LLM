"""Ingestion layer: fetch raw Indian legal documents from license-clean sources.

Interface: :class:`~indianlegal_llm.ingestion.base.BaseIngestor`.
Skeleton stub: :class:`~indianlegal_llm.ingestion.stub.StubIngestor`.
"""

from .base import BaseIngestor
from .stub import StubIngestor

__all__ = ["BaseIngestor", "StubIngestor"]
