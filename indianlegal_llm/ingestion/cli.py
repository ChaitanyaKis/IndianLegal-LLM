"""Ingestion CLI — stream a real source to RawDocs + a provenance manifest.

    python -m indianlegal_llm.ingestion --source sc --limit 5
    python -m indianlegal_llm.ingestion --source hc --limit 20
    python -m indianlegal_llm.ingestion --source indian-kanoon --doc-ids 91938676,257876
    python -m indianlegal_llm.ingestion --source india-code --urls @acts.txt

Default source is the real Supreme Court dataset (`sc`). Writes one provenance
line per document to the manifest (default `data/source_manifest.jsonl`, which is
gitignored). The corpus itself is streamed from S3 / the web and never saved to
local disk (CLAUDE.md §5). Unlike the answering pipeline, this command does NOT
fall back to the stub — a missing extra or bad source surfaces as a clear error.
"""

from __future__ import annotations

import argparse
import sys

from .._io import enable_utf8_output
from ..config import Settings
from .manifest import stream_to_manifest
from .registry import SOURCES, get_ingestor


def _split_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated list, or @file (one item per line)."""
    if not value:
        return None
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="python -m indianlegal_llm.ingestion",
        description="Stream Indian legal documents from a license-clean source.",
    )
    parser.add_argument(
        "--source",
        default="sc",
        help=f"source to ingest (default: sc). One of: {', '.join(SOURCES)} "
        "(aliases: sc, hc, kanoon, ik).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="max documents to pull (default: 200). Use a small value to sample.",
    )
    parser.add_argument(
        "--manifest",
        default=settings.manifest_path,
        help=f"manifest output path (default: {settings.manifest_path}).",
    )
    parser.add_argument(
        "--urls",
        default=None,
        help="india-code only: comma-separated act URLs, or @file.",
    )
    parser.add_argument(
        "--doc-ids",
        default=None,
        help="indian-kanoon only: comma-separated document ids, or @file.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="print a summary of the first N documents (default: 5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    args = _build_parser().parse_args(argv)
    if args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 2

    kwargs: dict = {}
    try:
        urls = _split_list(args.urls)
        doc_ids = _split_list(args.doc_ids)
    except OSError as exc:
        print(f"error: could not read list file: {exc}", file=sys.stderr)
        return 2
    if urls is not None:
        kwargs["urls"] = urls
    if doc_ids is not None:
        kwargs["doc_ids"] = doc_ids

    try:
        ingestor = get_ingestor(args.source, limit=args.limit, **kwargs)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Ingesting source={ingestor.source_name} limit={args.limit} ...")
    written = 0
    samples = []
    try:
        for doc in stream_to_manifest(ingestor.fetch(), args.manifest):
            written += 1
            if len(samples) < args.show:
                samples.append(doc)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # network / data error — report, do not fall back
        print(f"error while ingesting from {ingestor.source_name}: {exc}", file=sys.stderr)
        return 4

    print(f"\nWrote {written} document(s) to manifest: {args.manifest}\n")
    for doc in samples:
        print(f"- {doc.doc_id} | {doc.court} | {doc.date or '(no date)'} | {doc.language}")
        print(f"    {doc.url}")
        print(f"    {len(doc.text)} chars of text")
    if written == 0:
        print("(no documents matched — check --source/--limit and your inputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
