"""Quality evaluation (GPU, real retriever + model/adapter) — NON-blocking.

Run as a module (in the cloud, on a GPU host):

    python -m indianlegal_llm.evaluation.quality [--json PATH]

This is the SECOND, separate eval tier. Unlike the deterministic gate
(``evaluation.harness``), it builds the REAL pipeline from the environment
(INGESTOR=aws-sc + LLM=transformers + optional LORA_ADAPTER) and measures quality
on the lawyer-verified ``quality`` golden cases:

- citation_accuracy : grounded, cited answers for should-answer cases,
- retrieval_hit_rate : the expected authority's document is retrieved,
- proposition_grounding : the expected proposition appears in a retrieved chunk,
- task_metrics : a hook for LegalBench/LawBench-style tasks (plug in in-cloud).

It is NON-DETERMINISTIC and needs a GPU, so it is **never a merge gate**: it
always exits 0 and emits a report. If the real model/corpus is unavailable it
falls back to the stub (and says so), so the command still runs offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..pipeline import Pipeline, build_pipeline
from ..rag.citation import is_quote_grounded
from .cases import quality_cases
from .golden import HEADLINE_STATUSES, GoldenCase
from .schema import EvalCase


def _proposition_grounded(pipeline: Pipeline, case: EvalCase) -> bool | None:
    """True if the expected proposition appears in a retrieved chunk."""
    if not case.expected_proposition or case.expected_proposition.startswith("TODO"):
        return None
    retrieved = pipeline.answerer.retriever.retrieve(
        case.question, pipeline.settings.top_k
    )
    return any(is_quote_grounded(case.expected_proposition, rc.chunk.text) for rc in retrieved)


def run(pipeline: Pipeline | None = None, cases: list[EvalCase] | None = None) -> dict:
    """Run the quality eval and return a metrics report (never raises to fail CI)."""
    pipeline = pipeline or build_pipeline()
    cases = cases if cases is not None else quality_cases()

    answer_cases = [c for c in cases if c.should_answer]
    grounded = 0
    hits = 0
    hit_eligible = 0
    prop_hits = 0
    prop_eligible = 0
    per_case = []

    for case in cases:
        answer = pipeline.answer(case.question)
        is_grounded = answer.is_grounded
        if case.should_answer and is_grounded:
            grounded += 1
        retrieval_hit = None
        if case.expect_doc_id is not None:
            hit_eligible += 1
            retrieved = pipeline.answerer.retriever.retrieve(
                case.question, pipeline.settings.top_k
            )
            retrieval_hit = case.expect_doc_id in {rc.chunk.doc_id for rc in retrieved}
            hits += int(retrieval_hit)
        prop_hit = _proposition_grounded(pipeline, case)
        if prop_hit is not None:
            prop_eligible += 1
            prop_hits += int(prop_hit)
        per_case.append(
            {
                "case_id": case.case_id,
                "expected_authority": case.expected_authority,
                "refused": answer.refused,
                "grounded": is_grounded,
                "citations": [c.reference for c in answer.citations],
                "retrieval_hit": retrieval_hit,
                "proposition_grounded": prop_hit,
            }
        )

    def _rate(num: int, den: int) -> float | None:
        return (num / den) if den else None

    return {
        "backend": pipeline.llm_backend,  # TransformersLLM (real) or StubLLM (fallback)
        "source": pipeline.source,
        "is_real_run": pipeline.llm_backend != "StubLLM",
        "num_cases": len(cases),
        "citation_accuracy": _rate(grounded, len(answer_cases)),
        "retrieval_hit_rate": _rate(hits, hit_eligible),
        "proposition_grounding": _rate(prop_hits, prop_eligible),
        # Hook for external task suites (run in-cloud; datasets not vendored here).
        "task_metrics": {
            "legalbench": "TODO: plug in LegalBench tasks (in-cloud, see docs)",
            "lawbench": "TODO: plug in LawBench tasks (in-cloud, see docs)",
        },
        "per_case": per_case,
    }


def _format(report: dict) -> str:
    lines = ["", "IndianLegal-LLM - quality eval (NON-blocking)", "=" * 60]
    lines.append(f"backend={report['backend']}  source={report['source']}  real_run={report['is_real_run']}")
    if not report["is_real_run"]:
        lines.append("(stub fallback: run on a GPU host with the model/ingestion extras for real metrics)")
    lines.append("-" * 60)
    lines.append(f"cases={report['num_cases']}")
    lines.append(f"citation_accuracy={report['citation_accuracy']}")
    lines.append(f"retrieval_hit_rate={report['retrieval_hit_rate']}")
    lines.append(f"proposition_grounding={report['proposition_grounding']}")
    lines.append(f"task_metrics={report['task_metrics']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Golden-set scoring (REAL citations, present in the live index)
#
# Unlike run() above (which scores the stub-corpus quality cases), this scores
# GoldenCase records whose expected citations are REAL judgments in the index,
# proposed by the retriever and human-verified (CLAUDE.md §4). Only the
# human_verified+ subset is trustworthy enough to headline; machine_bootstrapped is
# reported separately as directional; unverified seeds are counted but never scored.
# --------------------------------------------------------------------------- #
def _doc_matches(expected_id: str, chunk) -> bool:
    """True if a retrieved chunk belongs to ``expected_id`` (an INSC neutral
    citation). Tolerates the LocalSCIngestor 'sc-<nc>' doc_id and the bare nc."""
    nc = (getattr(chunk, "metadata", None) or {}).get("nc_display", "")
    return expected_id in (chunk.doc_id, nc) or f"sc-{expected_id}" == chunk.doc_id


def _citation_doc_matches(expected_id: str, citation) -> bool:
    doc_id = getattr(citation, "doc_id", None)
    return expected_id in (doc_id, getattr(citation, "neutral_citation", "")) or (
        f"sc-{expected_id}" == doc_id
    )


def _score_one_golden(case: GoldenCase, retrieved: list, answer) -> dict:
    """Score a single case. Refusal probes are correct iff the system refused (no
    fabricated citation); holding/issue cases score retrieval/citation/grounding."""
    chunks = [rc.chunk for rc in retrieved]
    chunk_ids = {c.chunk_id for c in chunks}
    out: dict = {
        "id": case.id,
        "case_type": case.case_type,
        "status": case.verification_status,
    }

    if case.case_type == "refusal":
        out["refused"] = bool(answer.refused)
        out["refusal_correct"] = bool(answer.refused)  # correct = declined, no cite
        out["fabricated_citation"] = (not answer.refused) and bool(answer.citations)
        return out

    # holding / issue: retrieval hit + citation accuracy + proposition grounding.
    if case.expected_doc_ids:
        out["retrieval_hit"] = any(
            _doc_matches(e, c) for c in chunks for e in case.expected_doc_ids
        )
        cits = list(answer.citations)
        # Citation is correct iff the answer is grounded, every emitted citation was
        # actually retrieved (the trust guard guarantees this), AND every cited doc
        # is one of the expected authorities.
        out["citation_correct"] = bool(
            answer.is_grounded
            and cits
            and all(c.chunk_id in chunk_ids for c in cits)
            and all(
                any(_citation_doc_matches(e, c) for e in case.expected_doc_ids)
                for c in cits
            )
        )
    if case.expected_holding:
        out["proposition_grounded"] = any(
            is_quote_grounded(case.expected_holding, c.text) for c in chunks
        )
    return out


def _mean_bool(values) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def _aggregate_golden(results: list[dict]) -> dict:
    hi = [r for r in results if r["case_type"] in ("holding", "issue")]
    refusals = [r for r in results if r["case_type"] == "refusal"]
    return {
        "n_cases": len(results),
        "n_holding_issue": len(hi),
        "n_refusal": len(refusals),
        "retrieval_hit_rate": _mean_bool(r.get("retrieval_hit") for r in hi),
        "citation_accuracy": _mean_bool(r.get("citation_correct") for r in hi),
        "proposition_grounding": _mean_bool(r.get("proposition_grounded") for r in hi),
        "refusal_correctness": _mean_bool(r.get("refusal_correct") for r in refusals),
    }


def score_golden_cases(
    pipeline: Pipeline, cases: list[GoldenCase], *, top_k: int | None = None
) -> dict:
    """Score a golden set against the live pipeline. Returns headline metrics (the
    human_verified+ subset), directional metrics (machine_bootstrapped), counts by
    verification_status, and per-case detail. ``unverified`` seeds are counted but
    never scored (they carry no confirmed citation)."""
    top_k = top_k or pipeline.settings.top_k
    scored: list[dict] = []
    status_counts: dict[str, int] = {}
    for case in cases:
        status_counts[case.verification_status] = (
            status_counts.get(case.verification_status, 0) + 1
        )
        if case.verification_status == "unverified":
            continue  # seeds: counted, never scored (no confirmed citation)
        retrieved = pipeline.answerer.retriever.retrieve(case.question, top_k)
        answer = pipeline.answer(case.question)
        scored.append(_score_one_golden(case, retrieved, answer))

    headline = [r for r in scored if r["status"] in HEADLINE_STATUSES]
    directional = [r for r in scored if r["status"] == "machine_bootstrapped"]
    return {
        "backend": pipeline.llm_backend,
        "is_real_run": pipeline.llm_backend != "StubLLM",
        "source": pipeline.source,
        "status_counts": status_counts,
        "headline": _aggregate_golden(headline),  # human_verified + lawyer_verified
        "directional": _aggregate_golden(directional),  # machine_bootstrapped
        "per_case": scored,
    }


def _metric_lines(agg: dict) -> list[str]:
    return [
        f"  retrieval_hit_rate    = {agg['retrieval_hit_rate']}",
        f"  citation_accuracy     = {agg['citation_accuracy']}",
        f"  proposition_grounding = {agg['proposition_grounding']}",
        f"  refusal_correctness   = {agg['refusal_correctness']}",
    ]


def format_golden_report(report: dict) -> str:
    lines = ["", "IndianLegal-LLM - golden eval (REAL citations)", "=" * 60]
    lines.append(
        f"backend={report['backend']}  source={report['source']}  "
        f"real_run={report['is_real_run']}"
    )
    lines.append(f"cases by verification_status: {report['status_counts']}")
    headline, directional = report["headline"], report["directional"]
    lines.append("-" * 60)
    lines.append(f"HEADLINE (human_verified+, n={headline['n_cases']}):")
    if headline["n_cases"]:
        lines.extend(_metric_lines(headline))
    else:
        lines.append(
            "  (none yet — populate data/eval/golden.jsonl with human-verified "
            "cases; the headline stays empty until a human confirms real citations)"
        )
    lines.append(
        f"DIRECTIONAL (machine_bootstrapped, n={directional['n_cases']}, "
        "pending verification):"
    )
    if directional["n_cases"]:
        lines.extend(_metric_lines(directional))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .._io import enable_utf8_output

    from .golden import load_golden_cases

    enable_utf8_output()
    parser = argparse.ArgumentParser(prog="python -m indianlegal_llm.evaluation.quality")
    parser.add_argument("--json", default=None, help="write the quality report JSON here")
    parser.add_argument(
        "--golden",
        default=None,
        help="score this golden JSONL (REAL citations) instead of the stub set",
    )
    args = parser.parse_args([] if argv is None else argv)

    if args.golden:
        pipeline = build_pipeline()
        report = score_golden_cases(pipeline, load_golden_cases(args.golden))
        print(format_golden_report(report))
    else:
        report = run()
        print(_format(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0  # NON-blocking: never a merge gate


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
