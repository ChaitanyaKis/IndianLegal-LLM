"""Local Supreme Court ingestor — the REAL retrieval text source.

Reads the cleaned, processed judgment text produced by the build-time
``data_pipeline`` (PDF -> text) from ``data/sc/processed/year=*.jsonl`` and yields
:class:`RawDoc` for retrieval. This REPLACES the parquet ``raw_html`` (which is only
page chrome, not the judgment body) as the indexed body, so retrieval finally
indexes real judgment text — and chunks it with the SAME processor the fine-tune
builder uses (train == inference).

This is the MIT inference path: it reads plain JSONL (json + stdlib) and does NOT
import ``data_pipeline`` or any PDF library, keeping AGPL out of the shipped tree.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..schemas import RawDoc
from .base import BaseIngestor

_DEFAULT_PROCESSED_DIR = "data/sc/processed"


class LocalSCIngestor(BaseIngestor):
    """Yields SC judgments from the locally-processed (PDF-extracted) corpus."""

    source_name = "local-sc"

    def __init__(self, processed_dir: str | None = None, limit: int = 200) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.processed_dir = Path(processed_dir or _DEFAULT_PROCESSED_DIR)
        self.limit = limit

    def fetch(self) -> Iterator[RawDoc]:
        files = sorted(self.processed_dir.glob("year=*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"no processed corpus at {self.processed_dir} (year=*.jsonl). Run "
                "`python -m data_pipeline.corpus` to extract the SC PDFs first."
            )
        yielded = 0
        for path in files:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if yielded >= self.limit:
                        return
                    rec = json.loads(line)
                    if not rec.get("text"):
                        continue
                    yield RawDoc(
                        doc_id=rec["doc_id"],
                        title=rec.get("title", ""),
                        court=rec.get("court", "Supreme Court of India"),
                        date=rec.get("date", ""),
                        url=rec.get("url", ""),
                        license=rec.get("license", "government-work"),
                        text=rec["text"],
                        language=rec.get("language", "en"),
                        metadata={
                            "dataset": "local-sc",
                            "nc_display": rec.get("nc_display", ""),
                            "cnr": rec.get("cnr", ""),
                            "citation": rec.get("citation", ""),
                            "year": rec.get("year", ""),
                        },
                    )
                    yielded += 1
