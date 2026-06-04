"""The evaluation case contract."""

from __future__ import annotations

from dataclasses import dataclass

# Allowed values for EvalCase.expect.
EXPECT_ANSWER = "answer"
EXPECT_REFUSE = "refuse"


@dataclass(frozen=True)
class EvalCase:
    """One scored expectation against the pipeline.

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
        citations. ``None`` means "any grounded citation counts".
    note:
        Free-form description of why this case exists.
    """

    case_id: str
    question: str
    expect: str
    expect_doc_id: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.expect not in (EXPECT_ANSWER, EXPECT_REFUSE):
            raise ValueError(
                f"expect must be {EXPECT_ANSWER!r} or {EXPECT_REFUSE!r}, "
                f"got {self.expect!r}"
            )
