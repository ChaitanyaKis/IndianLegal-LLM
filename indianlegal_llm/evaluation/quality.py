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


def main(argv: list[str] | None = None) -> int:
    from .._io import enable_utf8_output

    enable_utf8_output()
    parser = argparse.ArgumentParser(prog="python -m indianlegal_llm.evaluation.quality")
    parser.add_argument("--json", default=None, help="write the quality report JSON here")
    args = parser.parse_args([] if argv is None else argv)

    report = run()
    print(_format(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0  # NON-blocking: never a merge gate


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
