"""Deterministic evaluation GATE (CLAUDE.md §6) — the blocking merge gate.

Run as a module:

    python -m indianlegal_llm.evaluation.harness [--json PATH] [--summary PATH]

Stub-pinned, offline, fast. It enforces INVARIANTS on every PR and exits non-zero
if any breaks:
- hallucinations == 0,
- citation_accuracy >= threshold (default 1.0; env EVAL_CITATION_THRESHOLD),
- refusal on out-of-corpus questions (refusal_accuracy == 1.0),
- retrieval hit-rate == 1.0 on the seed corpus,
- citation-guard properties hold: metadata-only citations, retrieved-set filter,
  quote-grounding (verified by a built-in adversarial self-check).

The separate, NON-blocking QUALITY eval (GPU, real retriever + model/adapter)
lives in ``evaluation.quality`` and produces a report, not a gate.

``--json`` writes machine-readable metrics; ``--summary`` writes a Markdown
metrics-delta (vs ``baseline_metrics.json``) suitable for a PR/job summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..ingestion.stub import StubIngestor
from ..model.base import BaseLLM
from ..model.stub import StubLLM
from ..pipeline import Pipeline, build_pipeline
from ..rag.answerer import Answerer
from ..schemas import Answer
from .cases import CASES
from .schema import EXPECT_ANSWER, EXPECT_REFUSE, EvalCase

_BASELINE_PATH = Path(__file__).with_name("baseline_metrics.json")


def _citation_threshold() -> float:
    raw = os.getenv("EVAL_CITATION_THRESHOLD")
    return float(raw) if raw else 1.0


@dataclass
class CaseResult:
    case: EvalCase
    answer: Answer
    citation_ok: bool
    refusal_ok: bool
    hallucinated: bool
    retrieval_hit: bool | None  # None when not applicable (refuse / no expected doc)

    @property
    def passed(self) -> bool:
        if self.case.expect == EXPECT_ANSWER:
            return self.citation_ok and not self.hallucinated
        return self.refusal_ok and not self.hallucinated


def _score_case(pipeline: Pipeline, case: EvalCase) -> CaseResult:
    answer = pipeline.answer(case.question)
    answered = not answer.refused
    cited_doc_ids = {c.doc_id for c in answer.citations}

    citation_ok = False
    refusal_ok = False
    retrieval_hit: bool | None = None
    if case.expect == EXPECT_ANSWER:
        citation_ok = answer.is_grounded and (
            case.expect_doc_id is None or case.expect_doc_id in cited_doc_ids
        )
        if case.expect_doc_id is not None:
            retrieved = pipeline.answerer.retriever.retrieve(
                case.question, pipeline.settings.top_k
            )
            retrieval_hit = case.expect_doc_id in {rc.chunk.doc_id for rc in retrieved}
    else:  # EXPECT_REFUSE
        refusal_ok = answer.refused

    hallucinated = (answered and not answer.citations) or (
        case.expect == EXPECT_REFUSE and answered
    )
    return CaseResult(
        case=case,
        answer=answer,
        citation_ok=citation_ok,
        refusal_ok=refusal_ok,
        hallucinated=hallucinated,
        retrieval_hit=retrieval_hit,
    )


# --------------------------------------------------------------------------- #
# Guard-invariant self-check (the gate enforces these directly, not only pytest)
# --------------------------------------------------------------------------- #
class _ScriptedLLM(BaseLLM):
    """An LLM returning a fixed (adversarial) string, for the self-check probes."""

    model_id = "scripted-self-check"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, system: str, user: str) -> str:
        return self._output


def guard_self_check() -> tuple[bool, list[str]]:
    """Adversarially probe the citation guard on the stub corpus.

    Returns (ok, failures). Mirrors the trust property: retrieved-set filter,
    quote-grounding, metadata-only citations, and out-of-corpus refusal.
    """
    base = build_pipeline(ingestor=StubIngestor(), llm=StubLLM())
    retriever = base.answerer.retriever
    settings = base.settings
    question = "Is privacy a fundamental right in India?"
    retrieved = retriever.retrieve(question, settings.top_k)
    if not retrieved:
        return False, ["seed query retrieved nothing"]

    chunk = retrieved[0].chunk
    real_id = chunk.chunk_id
    verbatim = " ".join(chunk.text.split()[1:8])  # a verbatim span (skip leading no.)
    failures: list[str] = []

    def probe(output: str) -> Answer:
        return Answerer(
            retriever=retriever, llm=_ScriptedLLM(output), settings=settings
        ).answer(question)

    if not probe("Privacy is protected. [ghost-not-retrieved::9]").refused:
        failures.append("retrieved-set: a non-retrieved citation survived")
    if not probe(
        'The Court said "the moon is made entirely of green cheese". [' + real_id + "]"
    ).refused:
        failures.append("quote-grounding: an ungrounded quote survived")
    if not probe("Privacy is certainly a fundamental right; take my word for it.").refused:
        failures.append("no-citation: an uncited answer was not refused")

    fabricated = probe(
        f'In Fabricated v. Nobody, 9999 INSC 9999, the Court noted "{verbatim}". [{real_id}]'
    )
    if fabricated.refused or not fabricated.citations:
        failures.append("metadata-only: a genuinely grounded answer was refused")
    else:
        reference = fabricated.citations[0].reference
        if "9999 INSC 9999" in reference or "Fabricated" in reference:
            failures.append("metadata-only: fabricated text leaked into the citation")
        if fabricated.citations[0].title != chunk.title:
            failures.append("metadata-only: citation not built from chunk metadata")

    if not base.answer("What is the capital of France?").refused:
        failures.append("out-of-corpus: a non-Indian-law question was answered")

    return (not failures), failures


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def run(pipeline: Pipeline | None = None, cases: list[EvalCase] | None = None) -> dict:
    """Run the deterministic gate and return a metrics dict (also used by tests).

    Pins the offline :class:`StubIngestor` + :class:`StubLLM` so the gate is
    deterministic and network-/GPU-independent regardless of INGESTOR / LLM.
    """
    pipeline = pipeline or build_pipeline(ingestor=StubIngestor(), llm=StubLLM())
    cases = cases if cases is not None else CASES

    results = [_score_case(pipeline, c) for c in cases]
    answer_results = [r for r in results if r.case.expect == EXPECT_ANSWER]
    refuse_results = [r for r in results if r.case.expect == EXPECT_REFUSE]
    hit_results = [r for r in answer_results if r.retrieval_hit is not None]

    citation_accuracy = (
        sum(r.citation_ok for r in answer_results) / len(answer_results)
        if answer_results
        else 1.0
    )
    refusal_accuracy = (
        sum(r.refusal_ok for r in refuse_results) / len(refuse_results)
        if refuse_results
        else 1.0
    )
    retrieval_hit_rate = (
        sum(bool(r.retrieval_hit) for r in hit_results) / len(hit_results)
        if hit_results
        else 1.0
    )
    hallucinations = sum(r.hallucinated for r in results)
    guard_ok, guard_failures = guard_self_check()

    threshold = _citation_threshold()
    green = (
        hallucinations == 0
        and citation_accuracy >= threshold
        and refusal_accuracy >= 1.0
        and retrieval_hit_rate >= 1.0
        and guard_ok
    )

    return {
        "results": results,
        "num_cases": len(cases),
        "citation_accuracy": citation_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "retrieval_hit_rate": retrieval_hit_rate,
        "hallucinations": hallucinations,
        "guard_invariants_ok": guard_ok,
        "guard_failures": guard_failures,
        "citation_threshold": threshold,
        "green": green,
    }


_SCALAR_KEYS = (
    "num_cases",
    "citation_accuracy",
    "refusal_accuracy",
    "retrieval_hit_rate",
    "hallucinations",
    "guard_invariants_ok",
    "green",
)


def metrics_scalars(metrics: dict) -> dict:
    """The committable subset of metrics (no per-case result objects)."""
    return {k: metrics[k] for k in _SCALAR_KEYS}


def _load_baseline() -> dict | None:
    try:
        return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _format_report(metrics: dict) -> str:
    lines = ["", "IndianLegal-LLM - deterministic eval gate", "=" * 76]
    lines.append(f"{'case':26} {'expect':8} {'result':28} {'pass':5}")
    lines.append("-" * 76)
    for r in metrics["results"]:
        result = "refused" if r.answer.refused else "cited:" + ",".join(
            c.doc_id for c in r.answer.citations
        )
        lines.append(
            f"{r.case.case_id:26} {r.case.expect:8} {result:28} "
            f"{'PASS' if r.passed else 'FAIL':5}"
        )
    lines.append("-" * 76)
    lines.append(f"citation_accuracy={metrics['citation_accuracy']}")
    lines.append(f"refusal_accuracy={metrics['refusal_accuracy']}")
    lines.append(f"retrieval_hit_rate={metrics['retrieval_hit_rate']}")
    lines.append(f"hallucinations={metrics['hallucinations']}")
    lines.append(f"guard_invariants_ok={metrics['guard_invariants_ok']}")
    if metrics["guard_failures"]:
        for failure in metrics["guard_failures"]:
            lines.append(f"  guard FAIL: {failure}")
    lines.append("")
    lines.append("GATE: GREEN" if metrics["green"] else "GATE: NOT GREEN")
    return "\n".join(lines)


def _markdown_summary(metrics: dict, baseline: dict | None) -> str:
    cur = metrics_scalars(metrics)
    rows = ["## IndianLegal-LLM — deterministic eval gate", ""]
    rows.append("| metric | baseline | current | delta |")
    rows.append("| --- | --- | --- | --- |")
    for key in _SCALAR_KEYS:
        current = cur[key]
        base_val = baseline.get(key) if baseline else None
        delta = ""
        if isinstance(current, (int, float)) and isinstance(base_val, (int, float)):
            d = current - base_val
            delta = "0" if d == 0 else f"{d:+g}"
        rows.append(f"| {key} | {base_val if base_val is not None else '-'} | {current} | {delta} |")
    rows.append("")
    if metrics["guard_failures"]:
        rows.append("**Guard invariant failures:**")
        rows.extend(f"- {f}" for f in metrics["guard_failures"])
        rows.append("")
    rows.append("**GATE: GREEN ✅**" if metrics["green"] else "**GATE: NOT GREEN ❌**")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    from .._io import enable_utf8_output

    enable_utf8_output()
    parser = argparse.ArgumentParser(prog="python -m indianlegal_llm.evaluation.harness")
    parser.add_argument("--json", default=None, help="write machine-readable metrics JSON here")
    parser.add_argument("--summary", default=None, help="write a Markdown metrics-delta here")
    # argv is None when called programmatically (e.g. tests) -> parse no args;
    # the CLI entrypoint passes sys.argv[1:] explicitly.
    args = parser.parse_args([] if argv is None else argv)

    metrics = run()
    print(_format_report(metrics))

    if args.json:
        Path(args.json).write_text(
            json.dumps(metrics_scalars(metrics), indent=2), encoding="utf-8"
        )
    if args.summary:
        Path(args.summary).write_text(
            _markdown_summary(metrics, _load_baseline()) + "\n", encoding="utf-8"
        )
    return 0 if metrics["green"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
