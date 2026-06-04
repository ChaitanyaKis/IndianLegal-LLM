"""The evaluation cases.

Two in-domain Indian-law questions that must be answered WITH a citation to the
right judgment, and one out-of-domain question that must be REFUSED (it exercises
the trust property in CLAUDE.md §4).
"""

from __future__ import annotations

from .schema import EXPECT_ANSWER, EXPECT_REFUSE, EvalCase

CASES: list[EvalCase] = [
    EvalCase(
        case_id="privacy",
        question="Is privacy a fundamental right in India?",
        expect=EXPECT_ANSWER,
        expect_doc_id="puttaswamy-2017",
        note="Puttaswamy: privacy is a fundamental right under Article 21.",
    ),
    EvalCase(
        case_id="basic-structure",
        question="What is the basic structure doctrine of the Indian Constitution?",
        expect=EXPECT_ANSWER,
        expect_doc_id="kesavananda-1973",
        note="Kesavananda: Parliament cannot destroy the basic structure.",
    ),
    EvalCase(
        case_id="capital-of-france",
        question="What is the capital of France?",
        expect=EXPECT_REFUSE,
        note="Out of domain (not Indian law): must refuse, no ungrounded claim.",
    ),
    EvalCase(
        case_id="european-union-structure",
        question="What is the structure of the European Union?",
        expect=EXPECT_REFUSE,
        note=(
            "Adversarial out-of-domain: shares only the incidental token "
            "'structure' with the corpus. The retriever's relevance gate must "
            "still drop it so the Answerer refuses (CLAUDE.md §1, §4)."
        ),
    ),
    EvalCase(
        case_id="us-bill-of-rights",
        question="What rights does the United States Bill of Rights guarantee?",
        expect=EXPECT_REFUSE,
        note=(
            "Adversarial out-of-domain: shares only 'rights'/'right' with the "
            "corpus. Must refuse rather than answer about foreign law."
        ),
    ),
]
