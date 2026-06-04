"""Ingestion layer: fetch raw Indian legal documents from license-clean sources.

Interface: :class:`~indianlegal_llm.ingestion.base.BaseIngestor`.

Stubs/implementations:
- :class:`~indianlegal_llm.ingestion.stub.StubIngestor` — offline, 2 SC judgments.
- AWS Open Data (real): ``aws-sc`` (Supreme Court), ``aws-hc`` (High Courts).
- Web (real): ``india-code`` (bare acts), ``indian-kanoon`` (judgments).

Use :func:`get_ingestor` to construct one by name; the real ingestors and their
optional dependencies are imported lazily, so importing this package needs only
the standard library. See the CLI: ``python -m indianlegal_llm.ingestion``.
"""

from .base import BaseIngestor
from .manifest import stream_to_manifest, write_manifest
from .registry import SOURCES, get_ingestor
from .stub import StubIngestor

__all__ = [
    "BaseIngestor",
    "StubIngestor",
    "get_ingestor",
    "SOURCES",
    "write_manifest",
    "stream_to_manifest",
]
