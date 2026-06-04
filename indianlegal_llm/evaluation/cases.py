"""Load the golden evaluation set from ``golden_set.json``.

The golden set is the single source of truth for both eval tiers. Inert TODO
placeholders (``verified_by`` starting with "TODO") are skipped. ``CASES`` is the
deterministic tier (the stub-pinned blocking gate); ``quality_cases()`` is the
GPU/real-pipeline tier.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import (
    EXPECT_ANSWER,
    EXPECT_REFUSE,
    TIER_DETERMINISTIC,
    TIER_QUALITY,
    EvalCase,
)

_GOLDEN_SET_PATH = Path(__file__).with_name("golden_set.json")


def load_golden_set(path: Path | None = None) -> list[EvalCase]:
    """Parse all golden-set cases (including inert TODO placeholders)."""
    data = json.loads((path or _GOLDEN_SET_PATH).read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for record in data["cases"]:
        cases.append(
            EvalCase(
                case_id=record["case_id"],
                question=record["question"],
                expect=EXPECT_ANSWER if record.get("should_answer") else EXPECT_REFUSE,
                expect_doc_id=record.get("expected_doc_id"),
                expected_authority=record.get("expected_authority", ""),
                expected_proposition=record.get("expected_proposition", ""),
                expected_pinpoint=record.get("expected_pinpoint", ""),
                verified_by=record.get("verified_by", ""),
                tiers=tuple(record.get("tiers", (TIER_DETERMINISTIC,))),
                note=record.get("note", ""),
            )
        )
    return cases


def _active(tier: str) -> list[EvalCase]:
    """Verified (non-TODO) cases for a given tier."""
    return [c for c in load_golden_set() if tier in c.tiers and not c.is_pending]


def deterministic_cases() -> list[EvalCase]:
    """Stub-pinned blocking-gate cases."""
    return _active(TIER_DETERMINISTIC)


def quality_cases() -> list[EvalCase]:
    """GPU/real-pipeline non-blocking cases."""
    return _active(TIER_QUALITY)


# Backward-compatible default: the deterministic tier drives the gate + tests.
CASES: list[EvalCase] = deterministic_cases()
