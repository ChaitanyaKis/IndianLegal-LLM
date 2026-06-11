#!/usr/bin/env python3
"""Propose REAL retrieval candidates for the golden eval set (human review step).

Build-time / cloud tool. Reads the seed queries (data/eval/golden_queries.yaml),
loads a PERSISTED e5 index (the embeddings.npy + chunks.jsonl that
scripts/kaggle_run.py wrote), and for each query emits the top-k REAL hits to
data/eval/golden_candidates.jsonl for a human to review.

CRITICAL (CLAUDE.md §4 — citation-trust product): this script ONLY surfaces real
retrievals. It NEVER invents an expected answer or holding. A human reads the
candidates and writes the verified cases into data/eval/golden.jsonl.

It reuses the repo's own ``EmbeddingRetriever`` (the same class build_pipeline
uses), reconstructed from the persisted index — retrieval ONLY, no LLM, CPU-ok
(it just encodes the short queries with e5-small). For refusal probes it records
the top score so a human can confirm it sits BELOW the refusal threshold (i.e. the
system would correctly refuse rather than fabricate).

Install: pip install -e .[rag,dataprep]   (sentence-transformers for the e5 query
encoder; pyyaml to read the seed queries). requirements.txt stays stdlib-only (§6).

Usage:
    python scripts/build_golden_candidates.py \
        --index-dir /kaggle/working/index \
        --queries data/eval/golden_queries.yaml \
        --out data/eval/golden_candidates.jsonl --top-k 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from indianlegal_llm._io import enable_utf8_output  # noqa: E402

_DEFAULT_INDEX_DIR = "/kaggle/working/index"
_DEFAULT_QUERIES = str(_REPO_ROOT / "data" / "eval" / "golden_queries.yaml")
_DEFAULT_OUT = str(_REPO_ROOT / "data" / "eval" / "golden_candidates.jsonl")


def _log(msg: str) -> None:
    print(f"[build_golden_candidates] {msg}", flush=True)


def _snippet(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _load_queries(path: Path) -> list[dict]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise SystemExit(
            "pyyaml is required to read the seed queries. "
            "Install the dataprep extra: pip install -e .[dataprep]"
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries", [])
    out = []
    for q in queries:
        if not q.get("query"):
            continue
        out.append({
            "id": q.get("id", ""),
            "query": q["query"],
            "case_type": q.get("case_type", "issue"),
        })
    return out


def _load_retriever(index_dir: Path):
    """Reconstruct the repo's EmbeddingRetriever from a persisted index.

    The min_score gate is turned OFF here so the raw top-k (incl. hits that would be
    below the production threshold) is visible for review — essential for refusal
    probes, where a human needs to see that the best score is BELOW the threshold.
    """
    import numpy as np

    from indianlegal_llm.rag.embedding_retriever import EmbeddingRetriever
    from indianlegal_llm.schemas import Chunk

    matrix_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"
    if not matrix_path.exists() or not chunks_path.exists():
        raise SystemExit(
            f"no persisted index at {index_dir} (need embeddings.npy + chunks.jsonl). "
            "Build one with scripts/kaggle_run.py first."
        )

    matrix = np.load(matrix_path)
    chunks: list[Chunk] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chunks.append(Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                text=r.get("text", ""),
                title=r.get("title", ""),
                court=r.get("court", ""),
                url=r.get("url", ""),
                license=r.get("license", ""),
                metadata={
                    "para_start": r.get("para_start"),
                    "para_end": r.get("para_end"),
                    "nc_display": r.get("nc_display", ""),
                    "year": r.get("year", ""),
                },
            ))
    if matrix.shape[0] != len(chunks):
        raise SystemExit(
            f"index mismatch: {matrix.shape[0]} vectors vs {len(chunks)} chunks "
            f"in {index_dir} (row i of the matrix must be chunk i)."
        )

    retriever = EmbeddingRetriever(min_score=0.0)  # gate off: see raw top-k scores
    retriever._chunks = chunks
    retriever._matrix = matrix.astype("float32")
    retriever.ensure_loaded()  # load e5-small for QUERY encoding only (CPU-ok)
    return retriever


def _production_threshold(index_dir: Path) -> float | None:
    """The refusal threshold (min_score) the index was built with, from its manifest."""
    manifest = index_dir / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return float(json.loads(manifest.read_text(encoding="utf-8")).get("min_score"))
    except (ValueError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    p = argparse.ArgumentParser(prog="python scripts/build_golden_candidates.py")
    p.add_argument("--index-dir", default=_DEFAULT_INDEX_DIR,
                   help="persisted e5 index dir")
    p.add_argument("--queries", default=_DEFAULT_QUERIES, help="seed queries YAML")
    p.add_argument("--out", default=_DEFAULT_OUT, help="candidates JSONL output")
    p.add_argument("--top-k", type=int, default=8,
                   help="candidates to surface per query")
    args = p.parse_args([] if argv is None else argv)

    index_dir = Path(args.index_dir)
    queries = _load_queries(Path(args.queries))
    if not queries:
        raise SystemExit(f"no queries found in {args.queries}")
    threshold = _production_threshold(index_dir)
    _log(f"{len(queries)} seed queries; refusal threshold (min_score) = {threshold}")

    retriever = _load_retriever(index_dir)
    _log(f"index: {len(retriever._chunks)} chunks loaded from {index_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for q in queries:
            hits = retriever.retrieve(q["query"], args.top_k)
            top_score = round(float(hits[0].score), 4) if hits else None
            records = []
            for rc in hits:
                chunk = rc.chunk
                meta = chunk.metadata or {}
                records.append({
                    "doc_id": chunk.doc_id,
                    "nc_display": meta.get("nc_display", ""),  # INSC neutral cite
                    "chunk_id": chunk.chunk_id,
                    "score": round(float(rc.score), 4),
                    "para_start": meta.get("para_start"),
                    "para_end": meta.get("para_end"),
                    "title": chunk.title,
                    "snippet": _snippet(chunk.text),
                })
            below = top_score is None or (
                threshold is not None and top_score < threshold
            )
            fh.write(json.dumps({
                "id": q["id"],
                "query": q["query"],
                "case_type": q["case_type"],
                "top_score": top_score,
                "refusal_threshold": threshold,
                # For refusal probes a human wants this True (best hit below threshold
                # -> the system would correctly refuse). For holding/issue it should
                # usually be False (something relevant was retrieved).
                "below_refusal_threshold": below,
                "candidates": records,  # REAL retrievals — a human picks the right one
            }, ensure_ascii=False) + "\n")
            n += 1
            flag = " [below threshold -> would refuse]" if below else ""
            _log(f"{q['id']:<28} top_score={top_score}{flag}")

    _log(f"wrote {n} query candidate sets to {out_path}")
    _log("NEXT: a human reviews these and writes VERIFIED cases into "
         "data/eval/golden.jsonl (never author citations from model knowledge).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
