"""Ingestion manifest writer (CLAUDE.md §3).

Writes one JSON object per line (JSONL) recording the provenance of every
ingested document: doc_id, url, court, date, language, license. The manifest
lives under the gitignored ``data/`` directory and is never committed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

from ..schemas import RawDoc


def write_manifest(docs: Iterable[RawDoc], path: str) -> int:
    """Write a provenance manifest line per doc; return the count written.

    The parent directory is created if needed. Returns the number of documents
    recorded so callers can report it.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.manifest_entry(), ensure_ascii=False) + "\n")
            count += 1
    return count


def stream_to_manifest(docs: Iterable[RawDoc], path: str) -> Iterator[RawDoc]:
    """Write each doc's manifest line as it passes through, yielding the doc.

    Lets a caller log provenance *while* streaming documents into processing,
    so a large corpus is never buffered in memory (CLAUDE.md §5).
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.manifest_entry(), ensure_ascii=False) + "\n")
            fh.flush()
            yield doc
