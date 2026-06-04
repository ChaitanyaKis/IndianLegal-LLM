"""The evaluation case contract (the golden-set record)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Allowed values for EvalCase.expect.
EXPECT_ANSWER = "answer"
EXPECT_REFUSE = "refuse"

# Evaluation tiers (kept separate, see docs/ROLES.md / harness vs quality):
TIER_DETERMINISTIC = "deterministic"  # stub-pinned, offline, blocking merge gate
TIER_QUALITY = "quality"  # GPU, real retriever+model, non-blocking metrics report


@dataclass(frozen=True)
class EvalCase:
    """One golden-set expectation against the pipeline.

    Attributes
    ----------
    case_id:
        Stable identifier for the case.
    question:
        The user question to ask the pipeline.
    expect:
        ``"answer"`` (a grounded, cited answer is expected) or ``"refuse"``.
    expect_doc_id:
        For ``expect == "answer"``, the doc_id that SHOULD appear among the
        citations on the *stub* corpus. ``None`` means "any grounded citation".
    expected_authority:
        Lawyer-facing: the neutral citation / case the answer should rest on
        (e.g. "2017 INSC 1"). Checked by the quality eval against the real corpus.
    expected_proposition:
        A short phrase the grounded answer should support (quality eval).
    expected_pinpoint:
        Expected paragraph pinpoint (e.g. "¶ 297"), lawyer-verified (quality eval).
    verified_by:
        Who verified this case. Cases whose ``verified_by`` starts with "TODO"
        are inert placeholders for a lawyer to fill in (skipped by both runners).
    tiers:
        Which eval tiers this case belongs to (``deterministic`` / ``quality``).
    note:
        Free-form description of why this case exists.
    """

    case_id: str
    question: str
    expect: str
    expect_doc_id: str | None = None
    expected_authority: str = ""
    expected_proposition: str = ""
    expected_pinpoint: str = ""
    verified_by: str = ""
    tiers: tuple[str, ...] = (TIER_DETERMINISTIC,)
    note: str = ""

    def __post_init__(self) -> None:
        if self.expect not in (EXPECT_ANSWER, EXPECT_REFUSE):
            raise ValueError(
                f"expect must be {EXPECT_ANSWER!r} or {EXPECT_REFUSE!r}, "
                f"got {self.expect!r}"
            )

    @property
    def should_answer(self) -> bool:
        return self.expect == EXPECT_ANSWER

    @property
    def is_pending(self) -> bool:
        """True for unverified TODO placeholders (a lawyer must fill these in)."""
        return self.verified_by.strip().upper().startswith("TODO")
