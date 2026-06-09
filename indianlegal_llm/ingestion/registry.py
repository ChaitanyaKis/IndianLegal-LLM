"""Map a source name to a configured ingestor.

All heavy/optional imports (pyarrow, s3fs, httpx) are deferred into each branch
so importing this module stays standard-library only. Used by both the ingestion
CLI and ``pipeline.build_pipeline()``.
"""

from __future__ import annotations

from .base import BaseIngestor
from .stub import StubIngestor

# Canonical source names (aliases accepted by get_ingestor).
SOURCES = ("stub", "local-sc", "aws-sc", "aws-hc", "india-code", "indian-kanoon")


def get_ingestor(source: str, limit: int = 200, **kwargs) -> BaseIngestor:
    """Return an ingestor for ``source``. Raises ValueError on an unknown name.

    Optional dependencies are imported lazily; a missing extra surfaces as an
    ImportError from the selected ingestor, not from importing this module.
    """
    name = (source or "stub").strip().lower()
    if name == "stub":
        return StubIngestor()
    if name in ("local-sc", "local"):
        from .local_sc import LocalSCIngestor

        return LocalSCIngestor(limit=limit)
    if name in ("aws-sc", "sc", "supreme-court"):
        from .aws_s3 import AWSSupremeCourtIngestor

        return AWSSupremeCourtIngestor(limit=limit, **kwargs)
    if name in ("aws-hc", "hc", "high-court"):
        from .aws_s3 import AWSHighCourtIngestor

        return AWSHighCourtIngestor(limit=limit, **kwargs)
    if name in ("india-code", "indiacode"):
        from .india_code import IndiaCodeIngestor

        return IndiaCodeIngestor(limit=limit, **kwargs)
    if name in ("indian-kanoon", "kanoon", "ik"):
        from .indian_kanoon import IndianKanoonIngestor

        return IndianKanoonIngestor(limit=limit, **kwargs)
    raise ValueError(
        f"unknown ingestor source {source!r}; choose from {', '.join(SOURCES)}"
    )
