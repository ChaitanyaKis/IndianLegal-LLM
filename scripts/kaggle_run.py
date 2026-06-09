#!/usr/bin/env python3
"""Kaggle-ready runner: embed + index + eval over the REAL SC judgment text.

This is a **build-time / cloud** tool (like ``data_pipeline``), not part of the
shipped framework. It runs on a Kaggle GPU host where the corpus is **not** local:
it streams the AWS Open Data Supreme Court tars from S3 one year at a time, reuses
the repo's existing extraction + chunking + retrieval, builds the multilingual-e5
vector index from the real judgment text, and runs the quality eval against it.

CLAUDE.md compliance
--------------------
- **§5 (cost/bandwidth):** the 42 GB corpus is **processed in the cloud and never
  kept whole**. Per-year streaming: sync one year's tar -> extract -> write the
  small processed JSONL -> **delete the raw tar before the next year**, so disk
  stays flat. Only small artifacts (the index, the coverage report, the eval
  results) land in ``/kaggle/working`` to come down — never the corpus.
- **§6 (green build):** the framework's ``requirements.txt`` stays stdlib-only;
  this script's heavy deps (sentence-transformers for e5, pyarrow/s3fs for S3,
  pdfminer.six/pypdf for extraction) come from the build-time extras
  (``pip install -e .[dataprep,rag,ingestion]`` + the AWS CLI), never from the
  shipped tree. Nothing here is imported by the MIT inference path.
- **Reuse, don't reimplement:** the index is built by ``build_pipeline()`` with
  ``INGESTOR=local-sc`` + ``VECTOR_BACKEND=e5``, so the SAME ``LocalSCIngestor`` +
  ``StubProcessor`` chunking + ``EmbeddingRetriever`` the inference path uses are
  what gets indexed (train == inference). The e5 model id is read from the repo's
  pinned default — never hardcoded here.

What it produces in ``--out-dir`` (default ``/kaggle/working``)
--------------------------------------------------------------
- ``coverage.csv`` / ``coverage.json`` — by-year extraction coverage table.
- ``index/`` — ``embeddings.npy`` (the e5 matrix), ``chunks.jsonl`` (chunk
  metadata + ¶ spans), ``manifest.json`` (model id, dims, counts, provenance).
- ``eval_real.json`` — the quality-eval report against the real-text index, plus
  ~10 holding/issue spot-check queries with their top hits (¶ + source).

Usage (on a Kaggle GPU notebook)
--------------------------------
    pip install -e .[dataprep,rag,ingestion] awscli
    python scripts/kaggle_run.py \
        --coverage-scope all \
        --embed-scope 2023-2025 \
        --llm transformers          # omit / use stub for a retrieval-only eval

On a multi-GPU host the embed pass shards encoding data-parallel across all CUDA
devices once the corpus clears ``--multi-gpu-threshold`` (default 50k chunks); e5
fits on one 16 GB GPU, so this is throughput, not model sharding. Extraction stays
on the CPU, independent of the GPU path. Run this AS A SCRIPT (the multi-process
pool uses the "spawn" start method, which needs this module's ``__main__`` guard);
do not import and call ``main()`` from a guard-less notebook cell.

Run ``python scripts/kaggle_run.py --help`` for every knob.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# data_pipeline lives at the repo root and is NOT pip-installed (pyproject
# packages.find includes only indianlegal_llm*), so put the repo root on the path
# before importing either package — works no matter the current directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_pipeline import corpus  # noqa: E402  (after sys.path shim)
from indianlegal_llm._io import enable_utf8_output  # noqa: E402
from indianlegal_llm.config import Settings  # noqa: E402
from indianlegal_llm.evaluation import quality  # noqa: E402
from indianlegal_llm.ingestion.local_sc import LocalSCIngestor  # noqa: E402
from indianlegal_llm.model.registry import get_llm  # noqa: E402
from indianlegal_llm.pipeline import build_pipeline  # noqa: E402
from indianlegal_llm.rag.embedding_retriever import (  # noqa: E402
    EmbeddingRetriever,
    cuda_device_count,
)

# The literal tar name data_pipeline.corpus.extract_year opens per year.
_TAR_NAME = "english.tar"

# AWS Open Data Supreme Court bucket (public, anonymous). See the ingestion layer
# (indianlegal_llm/ingestion/aws_s3.py) for the confirmed locations.
_DEFAULT_BUCKET = "indian-supreme-court-judgments"
_DEFAULT_REGION = "ap-south-1"
# The documented SC parquet/tar partition range (1950..2025). "all" expands here.
_FULL_SC_RANGE = "1950-2025"
# Default embed slice: the recent / Bharatiya-Nyaya-Sanhita-relevant era (BNS was
# enacted 2023, in force 1 Jul 2024). Tunable via --embed-scope.
_DEFAULT_EMBED_SCOPE = "2023-2025"

# Default LLM for the GPU quality eval: a small (3.8B) MIT instruct model that fits
# one T4 in 4-bit with headroom (phi-4 at ~14B is overkill for the eval and OOM'd).
# Verified MIT on its current HF card. Override with --llm-model / env LLM_MODEL
# (e.g. microsoft/phi-4). Also the model the loader FALLS BACK to if a bigger
# requested model fails to load — so the eval never silently drops to the stub.
_DEFAULT_EVAL_LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"

# ~10 holding/issue-style spot-check queries so retrieval relevance is eyeballable.
# Generic Indian-law questions (not pinned to any one judgment); override with
# --spot-queries FILE (one query per line).
_DEFAULT_SPOT_QUERIES = [
    "What did the Court hold on the right to privacy as a fundamental right?",
    "What is the test for granting anticipatory bail?",
    "When can a High Court quash an FIR under its inherent powers?",
    "What is the scope of judicial review of administrative action?",
    "What are the twin conditions for bail under the PMLA?",
    "What did the Court hold on reservation in promotions for public employment?",
    "When is a confession made to a police officer admissible in evidence?",
    "What is the doctrine of the basic structure of the Constitution?",
    "What are the rights of an arrested person under Article 22?",
    "What is the standard of proof for circumstantial evidence in a criminal trial?",
]


def _log(msg: str) -> None:
    print(f"[kaggle_run] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Scope parsing
# --------------------------------------------------------------------------- #
def _parse_scope(spec: str) -> list[str]:
    """Expand a scope spec into a sorted, de-duplicated list of year strings.

    Accepts the same forms as data_pipeline.corpus (``"2020"``, ``"2018-2026"``,
    ``"2018,2020"``) plus the alias ``"all"`` (-> the full documented SC range).
    """
    spec = (spec or "").strip()
    if spec.lower() == "all":
        spec = _FULL_SC_RANGE
    years = corpus._parse_years(spec)
    return sorted(set(years), key=int)


# --------------------------------------------------------------------------- #
# S3 sync (anonymous / --no-sign-request), bridging the S3<->local layout
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> int:
    """Run a subprocess, streaming nothing, returning its exit code."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        if tail:
            _log(f"  ! {cmd[0]} rc={proc.returncode}: {tail[-1]}")
    return proc.returncode


def _have_aws() -> bool:
    return shutil.which("aws") is not None


def _year_paths(sc_root: Path, year: str) -> tuple[Path, Path]:
    """(metadata dir, english-tar dir) in the LOCAL layout corpus.py reads."""
    return (
        sc_root / "metadata" / f"year={year}",
        sc_root / "data" / f"year={year}" / "english",
    )


def _normalize_tar(tar_dst: Path) -> None:
    """Ensure the tar is named ``english.tar`` (what extract_year opens).

    ``aws s3 sync`` / ``s3fs.get`` preserve the original S3 object name, which is
    not *guaranteed* to be ``english.tar``. If exactly one ``*.tar`` landed under
    a different name, rename it; if ``english.tar`` is already present, leave it.
    Anything else (zero or many non-``english.tar`` tars) is left for the caller's
    existence check to reject, rather than guessing.
    """
    target = tar_dst / _TAR_NAME
    if target.exists():
        return
    tars = sorted(tar_dst.glob("*.tar"))
    if len(tars) == 1:
        tars[0].rename(target)
    elif len(tars) > 1:
        _log(f"  ! {len(tars)} tars under {tar_dst} and none named {_TAR_NAME}; "
             "cannot disambiguate — treating year as unavailable.")


def _synced_ok(sc_root: Path, year: str) -> bool:
    """Success contract shared by BOTH backends: metadata.parquet AND the
    english tar are both present locally with the names extract_year requires."""
    meta_dst, tar_dst = _year_paths(sc_root, year)
    return (meta_dst / "metadata.parquet").exists() and (tar_dst / _TAR_NAME).exists()


def _sync_year_aws(bucket: str, region: str, year: str, sc_root: Path) -> bool:
    """Pull one year's metadata.parquet + english tar via the AWS CLI.

    Bridges the S3 layout (``metadata/parquet/...``, ``data/tar/...``) onto the
    LOCAL layout data_pipeline.corpus expects (no ``parquet/``/``tar/`` infix).
    Returns True only if both files actually landed (``aws s3 sync`` exits 0 even
    when it copies nothing, so the return code alone cannot be trusted).
    """
    meta_dst, tar_dst = _year_paths(sc_root, year)
    meta_dst.mkdir(parents=True, exist_ok=True)
    tar_dst.mkdir(parents=True, exist_ok=True)

    _run([
        "aws", "s3", "cp",
        f"s3://{bucket}/metadata/parquet/year={year}/metadata.parquet",
        str(meta_dst / "metadata.parquet"),
        "--no-sign-request", "--region", region, "--only-show-errors",
    ])
    _run([
        "aws", "s3", "sync",
        f"s3://{bucket}/data/tar/year={year}/english/",
        str(tar_dst),
        "--no-sign-request", "--region", region, "--only-show-errors",
        "--exclude", "*", "--include", "*.tar",
    ])
    _normalize_tar(tar_dst)
    return _synced_ok(sc_root, year)


def _sync_year_s3fs(bucket: str, region: str, year: str, sc_root: Path) -> bool:
    """Fallback pull via s3fs (anon=True) when the AWS CLI is unavailable.

    Same success contract as the AWS path (metadata + english tar both present).
    """
    try:
        import s3fs
    except ImportError:
        _log("  ! neither the AWS CLI nor s3fs is available; install one "
             "(pip install awscli  OR  pip install -e .[ingestion]).")
        return False

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": region})
    meta_dst, tar_dst = _year_paths(sc_root, year)
    meta_dst.mkdir(parents=True, exist_ok=True)
    tar_dst.mkdir(parents=True, exist_ok=True)

    meta_key = f"{bucket}/metadata/parquet/year={year}/metadata.parquet"
    tar_prefix = f"{bucket}/data/tar/year={year}/english/"
    try:
        if not fs.exists(meta_key):
            return False
        fs.get(meta_key, str(meta_dst / "metadata.parquet"))
        for key in fs.ls(tar_prefix):
            if str(key).endswith(".tar"):
                fs.get(str(key), str(tar_dst / Path(str(key)).name))
    except Exception as exc:  # noqa: BLE001 - report and let the caller skip
        _log(f"  ! s3fs sync failed for year={year}: {type(exc).__name__}: {exc}")
        return False
    _normalize_tar(tar_dst)
    return _synced_ok(sc_root, year)


def _sync_year(bucket: str, region: str, year: str, sc_root: Path) -> bool:
    if _have_aws():
        return _sync_year_aws(bucket, region, year, sc_root)
    return _sync_year_s3fs(bucket, region, year, sc_root)


def _delete_year_raw(sc_root: Path, year: str) -> None:
    """Delete the big tar (and its small metadata) for a year — disk stays flat."""
    for sub in (
        sc_root / "data" / f"year={year}",
        sc_root / "metadata" / f"year={year}",
    ):
        shutil.rmtree(sub, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Per-year extraction (stream -> extract -> write processed -> delete tar)
# --------------------------------------------------------------------------- #
def _coverage_sidecar(processed_dir: Path, year: str) -> Path:
    """Per-year Coverage record stored next to the JSONL, so re-runs report the
    real matched/extracted/quarantined numbers without re-downloading the tar."""
    return processed_dir / f"year={year}.coverage.json"


def _save_year_coverage(processed_dir: Path, cov: corpus.Coverage) -> None:
    _coverage_sidecar(processed_dir, cov.year).write_text(
        json.dumps(asdict(cov)), encoding="utf-8"
    )


def _load_year_coverage(processed_dir: Path, year: str) -> corpus.Coverage:
    """Restore a cached year's Coverage from its sidecar; fall back to counting
    emitted lines (older runs without a sidecar) so the table is never blank."""
    sidecar = _coverage_sidecar(processed_dir, year)
    if sidecar.exists():
        try:
            return corpus.Coverage(**json.loads(sidecar.read_text(encoding="utf-8")))
        except (ValueError, TypeError):
            pass
    out_path = processed_dir / f"year={year}.jsonl"
    n = (sum(1 for ln in out_path.open(encoding="utf-8") if ln.strip())
         if out_path.exists() else 0)
    return corpus.Coverage(year=year, emitted=n)


def _ensure_year_processed(
    year: str,
    *,
    bucket: str,
    region: str,
    sc_root: Path,
    processed_dir: Path,
    limit_per_year: int | None,
    allow_agpl: bool,
    force: bool = False,
) -> corpus.Coverage:
    """Stream + extract one year if its processed JSONL is missing.

    Returns the year's Coverage (fresh, or restored from the sidecar when the
    JSONL already exists). The raw tar is ALWAYS deleted before returning, so disk
    stays flat (CLAUDE.md §5). A year is only cached (JSONL written) when its tar
    truly arrived and extraction ran to completion — a failed/absent sync or an
    extraction error writes nothing, so the year is retried on the next run rather
    than poisoned into a permanently-empty cache.
    """
    out_path = processed_dir / f"year={year}.jsonl"
    if out_path.exists() and not force:
        _log(f"year={year}: processed JSONL already present — skipping S3 sync.")
        return _load_year_coverage(processed_dir, year)

    _log(f"year={year}: syncing from s3://{bucket} ...")
    try:
        if not _sync_year(bucket, region, year, sc_root):
            _log(f"year={year}: not available on S3 (or sync failed) — skipping.")
            return corpus.Coverage(year=year)
        try:
            records, cov = corpus.extract_year(
                sc_root, year, limit=limit_per_year, allow_agpl=allow_agpl
            )
        except Exception as exc:  # noqa: BLE001 - one bad year must not abort the run
            _log(f"year={year}: extraction failed ({type(exc).__name__}: {exc}) "
                 "— skipping (will retry next run).")
            return corpus.Coverage(year=year)
        # The tar was present (success contract), so an empty result is a genuine
        # "this year yielded no extractable text", not a missing-tar artifact —
        # write it so the year is not re-pulled, and record its coverage sidecar.
        corpus.write_processed(records, out_path)
        _save_year_coverage(processed_dir, cov)
        _log(
            f"year={year}: rows={cov.rows} matched={cov.matched_pdf} "
            f"extracted={cov.extracted} quarantined={cov.quarantined_scanned} "
            f"missing_pdf={cov.missing_pdf} emitted={cov.emitted}"
        )
        return cov
    finally:
        # Delete the raw tar before the next year regardless of outcome.
        _delete_year_raw(sc_root, year)


def _coverage_pct(cov: corpus.Coverage) -> float:
    return (cov.extracted / cov.matched_pdf * 100.0) if cov.matched_pdf else 0.0


def run_coverage(
    years: list[str],
    *,
    bucket: str,
    region: str,
    sc_root: Path,
    processed_dir: Path,
    out_dir: Path,
    limit_per_year: int | None,
    allow_agpl: bool,
) -> list[corpus.Coverage]:
    """Extraction-only pass over ``years``; emit the by-year coverage table."""
    _log(f"=== COVERAGE pass over {len(years)} year(s): {years[0]}..{years[-1]} ===")
    rows: list[corpus.Coverage] = []
    total = corpus.Coverage(year="ALL")
    for year in years:
        # _ensure_year_processed returns an accurate Coverage whether the year was
        # freshly extracted or restored from its sidecar on a re-run.
        cov = _ensure_year_processed(
            year, bucket=bucket, region=region, sc_root=sc_root,
            processed_dir=processed_dir, limit_per_year=limit_per_year,
            allow_agpl=allow_agpl,
        )
        rows.append(cov)
        total.merge(cov)

    _write_coverage(rows, total, out_dir)
    return rows


def _write_coverage(
    rows: list[corpus.Coverage], total: corpus.Coverage, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "coverage.csv"
    json_path = out_dir / "coverage.json"

    fields = [
        "year", "total_docs", "matched_pdf", "extracted", "quarantined_scans",
        "missing_pdf", "no_id", "emitted", "coverage_pct",
    ]

    def _row(cov: corpus.Coverage) -> dict:
        return {
            "year": cov.year,
            "total_docs": cov.rows,
            "matched_pdf": cov.matched_pdf,
            "extracted": cov.extracted,
            "quarantined_scans": cov.quarantined_scanned,
            "missing_pdf": cov.missing_pdf,
            "no_id": cov.no_id,  # rows with no derivable doc_id (can't be cited)
            "emitted": cov.emitted,
            "coverage_pct": round(_coverage_pct(cov), 1),
        }

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cov in rows:
            writer.writerow(_row(cov))
        writer.writerow(_row(total))

    json_path.write_text(
        json.dumps(
            {"by_year": [_row(c) for c in rows], "total": _row(total)},
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(
        f"coverage: matched_pdf={total.matched_pdf} extracted={total.extracted} "
        f"({_coverage_pct(total):.1f}% of matched) "
        f"quarantined_scans={total.quarantined_scanned} -> {csv_path}"
    )


# --------------------------------------------------------------------------- #
# Embed + index (reuse build_pipeline: local-sc + e5 + StubProcessor chunking)
# --------------------------------------------------------------------------- #
def _stage_embed_corpus(
    years: list[str],
    *,
    bucket: str,
    region: str,
    sc_root: Path,
    processed_dir: Path,
    embed_dir: Path,
    limit_per_year: int | None,
    allow_agpl: bool,
) -> int:
    """Make sure each embed-scope year is extracted, then collect just those
    years' processed JSONL into ``embed_dir`` so LocalSCIngestor sees only them.

    Returns the number of judgment records staged.
    """
    if embed_dir.exists():
        shutil.rmtree(embed_dir, ignore_errors=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    staged = 0
    for year in years:
        # Extract on demand (no-op if the coverage pass already produced it).
        _ensure_year_processed(
            year, bucket=bucket, region=region, sc_root=sc_root,
            processed_dir=processed_dir, limit_per_year=limit_per_year,
            allow_agpl=allow_agpl,
        )
        src = processed_dir / f"year={year}.jsonl"
        if not src.exists():
            _log(f"embed: year={year} has no processed JSONL — skipping.")
            continue
        n = sum(1 for line in src.open(encoding="utf-8") if line.strip())
        if n == 0:
            continue
        shutil.copy2(src, embed_dir / f"year={year}.jsonl")
        staged += n
    return staged


def _resolve_llm_model(flag: str | None, env_value: str | None) -> str:
    """Model-id precedence: --llm-model flag > env LLM_MODEL > small eval default."""
    return (flag or "").strip() or (env_value or "").strip() or _DEFAULT_EVAL_LLM_MODEL


def _default_llm_loader(model_id: str, adapter: str):
    """Construct + eagerly load a real transformers LLM (raises if it can't load)."""
    llm = get_llm("transformers", base_model=model_id, adapter=adapter)
    ensure = getattr(llm, "ensure_loaded", None)
    if ensure is not None:
        ensure()  # force the weight load now so a failure surfaces here, not later
    return llm


def _load_real_llm(primary: str, fallback: str, *, adapter: str = "", loader=None):
    """Load a REAL LLM, falling back to a smaller model before ever giving up.

    Tries ``primary``; on any failure logs a prominent FALLING BACK message and
    tries ``fallback`` (skipped if identical). If both fail it RAISES — it never
    returns a StubLLM, so the eval can never silently report stub metrics as real.
    """
    loader = loader or _default_llm_loader
    attempts = [primary] + ([fallback] if fallback and fallback != primary else [])
    last_err = ""
    for i, model_id in enumerate(attempts):
        try:
            llm = loader(model_id, adapter)
            if i > 0:
                _log(f"⚠ FELL BACK to '{model_id}' for the eval LLM.")
            else:
                _log(f"eval LLM: loaded '{model_id}'.")
            return llm
        except Exception as exc:  # noqa: BLE001 - try the fallback, then surface
            last_err = f"{type(exc).__name__}: {exc}"
            tag = "primary" if i == 0 else "fallback"
            _log(f"⚠ eval LLM {tag} '{model_id}' failed to load: {last_err}")
            if i == 0 and len(attempts) > 1:
                _log(f"⚠ FALLING BACK to '{fallback}': {last_err}")
    raise RuntimeError(
        f"could not load a real eval LLM (tried {attempts}; last error: {last_err}). "
        "Refusing to fall back to the stub and report its metrics as a real run — "
        "fix the model/GPU/deps (pip install -e .[model]) or use --llm stub."
    )


def run_embed_index(
    years: list[str],
    *,
    bucket: str,
    region: str,
    sc_root: Path,
    processed_dir: Path,
    embed_dir: Path,
    out_dir: Path,
    llm: str,
    llm_model: str,
    adapter: str,
    top_k: int,
    limit_per_year: int | None,
    embed_limit: int,
    allow_agpl: bool,
    multi_gpu_threshold: int,
):
    """Build the e5 vector index over the embed-scope real text and persist it.

    Reuses ``build_pipeline`` so the indexed chunks come from the exact same
    LocalSCIngestor + StubProcessor + EmbeddingRetriever the inference path uses.
    Returns the built Pipeline (so the eval pass can query the same index).
    """
    _log(f"=== EMBED+INDEX pass over {len(years)} year(s): {years} ===")
    staged = _stage_embed_corpus(
        years, bucket=bucket, region=region, sc_root=sc_root,
        processed_dir=processed_dir, embed_dir=embed_dir,
        limit_per_year=limit_per_year, allow_agpl=allow_agpl,
    )
    if staged == 0:
        raise RuntimeError(
            f"no processed judgments staged for embed-scope {years} — nothing to "
            "index (check the years are on S3 and extraction produced text)."
        )

    # Preflight the rag extra BEFORE build_pipeline: _resolve_retriever silently
    # degrades to the lexical retriever if sentence-transformers is missing, and
    # build_pipeline would still go on to load the (heavy) LLM. Fail fast here so a
    # missing rag extra never burns a GPU model load only to be caught below.
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "the e5 index needs sentence-transformers; install the rag extra "
            "(pip install -e .[rag]) before running the embed pass."
        ) from exc

    # Multi-GPU data-parallel embedding: e5 fits on one 16 GB GPU, so this is pure
    # throughput sharding. The threshold is read by MultilingualE5Embedder from the
    # env at construction (inside build_pipeline), so set it BEFORE building. The
    # embedder logs which path (single- vs multi-device) actually ran.
    os.environ["E5_MULTI_GPU_MIN_CHUNKS"] = str(multi_gpu_threshold)
    n_gpu = cuda_device_count()
    if n_gpu > 1 and multi_gpu_threshold > 0:
        _log(f"embed: {n_gpu} CUDA devices visible; data-parallel embedding engages "
             f"at >={multi_gpu_threshold} chunks (else single-device).")
    else:
        reason = ("multi-GPU disabled (threshold<=0)" if multi_gpu_threshold <= 0
                  else f"{n_gpu} CUDA device(s) visible")
        _log(f"embed: single-device embedding — {reason}.")

    # Resolve the eval LLM. For a real run we pre-load it here (with a loud
    # fallback to the small model) and hand the loaded instance to build_pipeline,
    # so build_pipeline's OWN silent stub fallback never fires — a real run that
    # can't load any model ERRORS instead of quietly reporting stub metrics. For
    # --llm stub we leave the (intentional) stub to build_pipeline via settings.
    real_llm = None
    if llm == "transformers":
        real_llm = _load_real_llm(
            llm_model, _DEFAULT_EVAL_LLM_MODEL, adapter=adapter
        )

    _log(f"embed: staged {staged} judgment(s); building the e5 index ...")

    # Reuse the sanctioned wiring point. VECTOR_BACKEND=e5 selects the real
    # EmbeddingRetriever; the ingestor override points it at exactly the embed
    # years. ``llm`` is passed pre-loaded for a real run (else resolved from
    # settings, i.e. the stub) — build_pipeline stays the single wiring point.
    settings = Settings(
        vector_backend="e5", llm=llm, base_model=llm_model, adapter=adapter, top_k=top_k
    )
    ingestor = LocalSCIngestor(processed_dir=str(embed_dir), limit=embed_limit)
    pipeline = build_pipeline(settings, ingestor=ingestor, llm=real_llm)

    retriever = pipeline.answerer.retriever
    if not isinstance(retriever, EmbeddingRetriever):
        raise RuntimeError(
            f"expected the e5 EmbeddingRetriever but got {type(retriever).__name__}. "
            "build_pipeline fell back to the lexical retriever — install the rag "
            "extra (pip install -e .[rag]) so sentence-transformers + e5 are present."
        )
    if pipeline.source != "local-sc":
        raise RuntimeError(
            f"expected the local-sc source but indexed '{pipeline.source}' — the "
            "real judgment text was not indexed (check the staged processed JSONL)."
        )

    _persist_index(pipeline, retriever, years, out_dir)
    return pipeline


def _persist_index(pipeline, retriever, years: list[str], out_dir: Path) -> dict:
    """Serialize the e5 matrix + chunk metadata + a manifest to ``out_dir/index``."""
    import numpy as np

    chunks = list(getattr(retriever, "_chunks", []))
    matrix = getattr(retriever, "_matrix", None)
    if matrix is None or not chunks:
        raise RuntimeError("the e5 retriever holds no embedded chunks to persist.")

    index_dir = out_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    np.save(index_dir / "embeddings.npy", np.asarray(matrix, dtype="float32"))

    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            meta = chunk.metadata or {}
            fh.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "court": chunk.court,
                "url": chunk.url,
                "license": chunk.license,
                "para_start": meta.get("para_start"),
                "para_end": meta.get("para_end"),
                "nc_display": meta.get("nc_display", ""),
                "year": meta.get("year", ""),
                "text": chunk.text,
            }, ensure_ascii=False) + "\n")

    embedder = getattr(retriever, "embedder", None)
    manifest = {
        "kind": "indianlegal-llm.e5-index",
        # The EXACT model id the repo pinned — read off the live embedder, not
        # hardcoded here (CLAUDE.md: adapt the implementation, not the contract).
        "embedding_model": getattr(embedder, "model_name", "unknown"),
        "e5_prefixes": {"passage": "passage: ", "query": "query: "},
        "min_score": float(getattr(retriever, "min_score", 0.0)),
        "dim": int(matrix.shape[1]),
        "num_chunks": int(matrix.shape[0]),
        "num_docs": len({c.doc_id for c in chunks}),
        "embed_scope_years": years,
        "source": pipeline.source,
        "chunker": "StubProcessor (paragraph-aware; same as retrieval/fine-tune)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": _git_commit(),
        "note": (
            "Reload: np.load('embeddings.npy') + read chunks.jsonl line-for-line "
            "(row i of the matrix is chunk i). Cosine = matrix @ e5_query_vec."
        ),
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _log(
        f"index: {manifest['num_chunks']} chunks / {manifest['num_docs']} docs, "
        f"dim={manifest['dim']}, model={manifest['embedding_model']} -> {index_dir}"
    )
    return manifest


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Eval (repo quality eval against the real index + manual spot-checks)
# --------------------------------------------------------------------------- #
def _snippet(text: str, n: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _pinpoint(meta: dict) -> str:
    start, end = meta.get("para_start"), meta.get("para_end")
    if start is None:
        return "¶ —"
    return f"¶ {start}" if (end is None or end == start) else f"¶ {start}-{end}"


def _raw_top_score(retriever, query: str) -> float | None:
    """Top-1 cosine BEFORE the min_score gate, so an empty result can be read as
    'gate too high' (e.g. 0.79 vs a 0.80 gate) rather than 'nothing relevant'."""
    matrix = getattr(retriever, "_matrix", None)
    embedder = getattr(retriever, "embedder", None)
    if matrix is None or embedder is None:
        return None
    try:
        import numpy as np
        q = np.asarray(embedder.embed_query(query), dtype="float32")
        return float((matrix @ q).max())
    except Exception:  # noqa: BLE001 - diagnostics only, never fail the eval
        return None


def run_spot_checks(pipeline, queries: list[str], top_k: int) -> list[dict]:
    """Print each query's top hits (¶ + source) and return them structured."""
    _log(f"=== SPOT-CHECK: {len(queries)} holding/issue queries (top_k={top_k}) ===")
    retriever = pipeline.answerer.retriever
    min_score = float(getattr(retriever, "min_score", 0.0))
    results: list[dict] = []
    for q in queries:
        hits = retriever.retrieve(q, top_k)
        top_raw = _raw_top_score(retriever, q)
        print(f"\nQ: {q}")
        if not hits:
            raw = f"{top_raw:.3f}" if top_raw is not None else "n/a"
            print(f"   (no chunk cleared the gate min_score={min_score:.2f}; "
                  f"best pre-gate cosine={raw} — would refuse)")
        hit_records = []
        for rank, rc in enumerate(hits, start=1):
            chunk = rc.chunk
            meta = chunk.metadata or {}
            nc = meta.get("nc_display", "")
            src = f"{chunk.title or chunk.doc_id}" + (f" [{nc}]" if nc else "")
            print(f"   {rank}. score={rc.score:.3f}  {_pinpoint(meta)}  {src}")
            print(f"      {chunk.doc_id} :: {chunk.chunk_id}")
            print(f"      {_snippet(chunk.text)}")
            hit_records.append({
                "rank": rank,
                "score": round(float(rc.score), 4),
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "neutral_citation": nc,
                "para_start": meta.get("para_start"),
                "para_end": meta.get("para_end"),
                "snippet": _snippet(chunk.text),
            })
        results.append({
            "query": q,
            "num_hits": len(hits),
            "top_pre_gate_cosine": round(top_raw, 4) if top_raw is not None else None,
            "hits": hit_records,
        })
    return results


def run_eval(pipeline, out_dir: Path, queries: list[str], top_k: int) -> dict:
    """Run the repo quality eval against the real index + the spot-checks."""
    _log("=== QUALITY eval (repo harness, real e5 index) ===")
    report = quality.run(pipeline=pipeline)
    print(quality._format(report))
    if not report.get("is_real_run"):
        # Unmissable banner: stub metrics must NEVER read as a real-model run.
        bar = "=" * 64
        print(
            f"\n{bar}\n"
            "⚠  EVAL DID NOT USE A REAL LLM  (backend = StubLLM)\n"
            f"{bar}\n"
            "   citation_accuracy below is the stub's deterministic behavior, NOT a\n"
            "   real model. Pass --llm transformers on a GPU host for a real run.\n"
            "   (retrieval_hit_rate/proposition_grounding + the spot-checks ARE real:\n"
            "    they come from the e5 retriever and are independent of the LLM.)\n"
            f"{bar}\n",
            file=sys.stderr,
        )

    # The golden set's expected_doc_ids (e.g. 'puttaswamy-2017') belong to the STUB
    # corpus; the local-sc corpus uses 'sc-<neutral-citation>' ids, so they can
    # never match and retrieval_hit_rate is structurally N/A here. Flag it rather
    # than letting a meaningless 0.0 read as a real quality signal. Real retrieval
    # quality on this corpus is read from proposition_grounding + the spot-checks.
    retriever = pipeline.answerer.retriever
    notes = {
        "retrieval_hit_rate_valid": pipeline.source != "local-sc",
        "retrieval_hit_rate_note": (
            "N/A for the local-sc corpus: golden expected_doc_ids target the stub "
            "corpus and never match 'sc-<...>' ids. Use proposition_grounding + "
            "the spot-checks for real-corpus retrieval quality."
            if pipeline.source == "local-sc" else
            "expected_doc_ids match this source; the value is meaningful."
        ),
        "e5_min_score": float(getattr(retriever, "min_score", 0.0)),
    }
    if pipeline.source == "local-sc":
        _log("note: retrieval_hit_rate is N/A for local-sc (golden expected_doc_ids "
             "target the stub corpus). Judge retrieval via proposition_grounding + "
             "the spot-checks' pre-gate cosine scores.")

    spot = run_spot_checks(pipeline, queries, top_k)

    out_dir.mkdir(parents=True, exist_ok=True)
    llm = pipeline.answerer.llm
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "is_real_run": bool(report.get("is_real_run")),
        "llm_backend": report.get("backend"),
        "llm_model": getattr(llm, "model_id", "unknown"),
        "quality_eval": report,
        "metric_notes": notes,
        "spot_checks": spot,
    }
    eval_path = out_dir / "eval_real.json"
    eval_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(f"eval: wrote {eval_path}")
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_queries(path: str | None) -> list[str]:
    if not path:
        return list(_DEFAULT_SPOT_QUERIES)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    queries = [
        ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")
    ]
    return queries or list(_DEFAULT_SPOT_QUERIES)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/kaggle_run.py",
        description="Kaggle runner: stream SC tars from S3, build the e5 index over "
                    "real judgment text, and eval it. Corpus never persists whole.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bucket", default=_DEFAULT_BUCKET, help="public SC S3 bucket")
    p.add_argument("--region", default=_DEFAULT_REGION, help="S3 region (anonymous)")
    p.add_argument("--coverage-scope", default="all",
                   help='extraction-only scope: "all", "2018-2026", "2020,2021"')
    p.add_argument("--embed-scope", default=_DEFAULT_EMBED_SCOPE,
                   help="years to embed+index this pass (recent/BNS era by default)")
    p.add_argument("--work-dir", default="/kaggle/working",
                   help="scratch + output root (Kaggle's writable dir)")
    p.add_argument("--sc-root", default=None,
                   help="where synced tars/parquet land (default <work-dir>/sc)")
    p.add_argument("--processed-dir", default=None,
                   help="processed year=*.jsonl dir (default <sc-root>/processed)")
    p.add_argument("--out-dir", default=None,
                   help="artifact output dir (default <work-dir>)")
    p.add_argument("--llm", default="stub", choices=["stub", "transformers"],
                   help="LLM for the quality eval; 'stub' keeps it retrieval-focused")
    p.add_argument("--llm-model", default=None,
                   help="eval LLM id (else env LLM_MODEL, else a small MIT instruct "
                        f"model: {_DEFAULT_EVAL_LLM_MODEL}). e.g. microsoft/phi-4")
    p.add_argument("--adapter", default="",
                   help="optional LoRA/QLoRA adapter (path or HF id) for the eval LLM")
    p.add_argument("--top-k", type=int, default=5, help="chunks retrieved per query")
    p.add_argument("--limit-per-year", type=int, default=None,
                   help="cap docs/year (smoke runs)")
    p.add_argument("--embed-limit", type=int, default=1_000_000,
                   help="max judgments to embed (safety cap)")
    p.add_argument("--multi-gpu-threshold", type=int, default=50_000,
                   help="min chunk count to shard embedding data-parallel across "
                        "all CUDA devices (needs >1 GPU); 0 disables multi-GPU")
    p.add_argument("--spot-queries", default=None,
                   help="file of newline-delimited spot-check queries (else built-in)")
    p.add_argument("--allow-agpl", action="store_true",
                   help="build-only PyMuPDF extraction fallback (off by default)")
    p.add_argument("--skip-coverage", action="store_true",
                   help="skip the all-years coverage pass")
    p.add_argument("--skip-embed", action="store_true",
                   help="skip the embed+index+eval pass")
    return p


def main(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    args = build_arg_parser().parse_args([] if argv is None else argv)

    if args.skip_coverage and args.skip_embed:
        _log("nothing to do: both --skip-coverage and --skip-embed were set.")
        return 2

    work_dir = Path(args.work_dir)
    sc_root = Path(args.sc_root) if args.sc_root else work_dir / "sc"
    processed_dir = (
        Path(args.processed_dir) if args.processed_dir else sc_root / "processed"
    )
    out_dir = Path(args.out_dir) if args.out_dir else work_dir
    embed_dir = work_dir / "embed_input"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not _have_aws():
        _log("note: AWS CLI not found; will try s3fs (pip install -e .[ingestion]).")

    if not args.skip_coverage:
        run_coverage(
            _parse_scope(args.coverage_scope),
            bucket=args.bucket, region=args.region, sc_root=sc_root,
            processed_dir=processed_dir, out_dir=out_dir,
            limit_per_year=args.limit_per_year, allow_agpl=args.allow_agpl,
        )

    if not args.skip_embed:
        llm_model = _resolve_llm_model(args.llm_model, os.getenv("LLM_MODEL"))
        pipeline = run_embed_index(
            _parse_scope(args.embed_scope),
            bucket=args.bucket, region=args.region, sc_root=sc_root,
            processed_dir=processed_dir, embed_dir=embed_dir, out_dir=out_dir,
            llm=args.llm, llm_model=llm_model, adapter=args.adapter,
            top_k=args.top_k, limit_per_year=args.limit_per_year,
            embed_limit=args.embed_limit, allow_agpl=args.allow_agpl,
            multi_gpu_threshold=args.multi_gpu_threshold,
        )
        run_eval(pipeline, out_dir, _load_queries(args.spot_queries), args.top_k)

    _log("done. Small artifacts are in the output dir; no corpus was kept whole.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
