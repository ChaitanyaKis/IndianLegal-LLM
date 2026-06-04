"""Evaluation harness — the green-build gate (CLAUDE.md §6).

Run as a module:

    python -m indianlegal_llm.evaluation.harness

It builds the pipeline via `build_pipeline()` (never wiring components itself),
scores every case, prints a per-case table plus the three headline metrics, and
exits non-zero if the build is not green.

Metrics
-------
citation_accuracy : fraction of "answer" cases that returned a grounded answer
                    citing the expected document.
refusal_accuracy  : fraction of "refuse" cases that actually refused.
hallucinations    : count of answers that assert a claim without grounding —
                    i.e. a non-refused answer with no citation, or answering a
                    question that should have been refused.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..ingestion.stub import StubIngestor
from ..model.stub import StubLLM
from ..pipeline import Pipeline, build_pipeline
from ..schemas import Answer
from .cases import CASES
from .schema import EXPECT_ANSWER, EXPECT_REFUSE, EvalCase


@dataclass
class CaseResult:
    case: EvalCase
    answer: Answer
    citation_ok: bool
    refusal_ok: bool
    hallucinated: bool

    @property
    def passed(self) -> bool:
        if self.case.expect == EXPECT_ANSWER:
            return self.citation_ok and not self.hallucinated
        return self.refusal_ok and not self.hallucinated


def _score_case(case: EvalCase, answer: Answer) -> CaseResult:
    answered = not answer.refused
    cited_doc_ids = {c.doc_id for c in answer.citations}

    citation_ok = False
    refusal_ok = False
    if case.expect == EXPECT_ANSWER:
        citation_ok = answer.is_grounded and (
            case.expect_doc_id is None or case.expect_doc_id in cited_doc_ids
        )
    else:  # EXPECT_REFUSE
        refusal_ok = answer.refused

    # A hallucination is any ungrounded assertion:
    #   * a non-refused answer carrying no citation, or
    #   * answering a question that should have been refused.
    hallucinated = (answered and not answer.citations) or (
        case.expect == EXPECT_REFUSE and answered
    )
    return CaseResult(
        case=case,
        answer=answer,
        citation_ok=citation_ok,
        refusal_ok=refusal_ok,
        hallucinated=hallucinated,
    )


def run(pipeline: Pipeline | None = None, cases: list[EvalCase] | None = None) -> dict:
    """Run the eval and return a metrics dict (also used by tests).

    The eval pins the offline :class:`StubIngestor` and :class:`StubLLM` so the
    green-build gate is deterministic and network-/GPU-independent regardless of
    the configured INGESTOR and LLM.
    """
    pipeline = pipeline or build_pipeline(ingestor=StubIngestor(), llm=StubLLM())
    cases = cases if cases is not None else CASES

    results = [_score_case(c, pipeline.answer(c.question)) for c in cases]

    answer_cases = [r for r in results if r.case.expect == EXPECT_ANSWER]
    refuse_cases = [r for r in results if r.case.expect == EXPECT_REFUSE]

    citation_accuracy = (
        sum(r.citation_ok for r in answer_cases) / len(answer_cases)
        if answer_cases
        else 1.0
    )
    refusal_accuracy = (
        sum(r.refusal_ok for r in refuse_cases) / len(refuse_cases)
        if refuse_cases
        else 1.0
    )
    hallucinations = sum(r.hallucinated for r in results)

    return {
        "results": results,
        "citation_accuracy": citation_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "hallucinations": hallucinations,
        "green": (
            citation_accuracy == 1.0
            and refusal_accuracy == 1.0
            and hallucinations == 0
        ),
    }


def _format_report(metrics: dict) -> str:
    # Keep every printed character ASCII so the report never raises
    # UnicodeEncodeError on a Windows OEM codepage (cp437/cp850) or redirected
    # stdout. Column widths are sized so the PASS/FAIL column stays aligned.
    lines = ["", "IndianLegal-LLM - evaluation harness", "=" * 76]
    header = f"{'case':26} {'expect':8} {'result':28} {'pass':5}"
    lines.append(header)
    lines.append("-" * 76)
    for r in metrics["results"]:
        if r.answer.refused:
            result = "refused"
        else:
            result = "cited:" + ",".join(c.doc_id for c in r.answer.citations)
        lines.append(
            f"{r.case.case_id:26} {r.case.expect:8} {result:28} "
            f"{'PASS' if r.passed else 'FAIL':5}"
        )
    lines.append("-" * 72)
    lines.append(f"citation_accuracy={metrics['citation_accuracy']}")
    lines.append(f"refusal_accuracy={metrics['refusal_accuracy']}")
    lines.append(f"hallucinations={metrics['hallucinations']}")
    lines.append("")
    lines.append("BUILD: GREEN" if metrics["green"] else "BUILD: NOT GREEN")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .._io import enable_utf8_output

    enable_utf8_output()
    metrics = run()
    print(_format_report(metrics))
    return 0 if metrics["green"] else 1


if __name__ == "__main__":
    sys.exit(main())
