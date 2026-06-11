"""Golden eval-set contract for IndianLegal-LLM — REAL, human-verified citations.

A :class:`GoldenCase` pins a question to the REAL Supreme Court judgment(s) that
answer it, as present in the live index (SC 2023–2025, multilingual-e5).

CRITICAL (CLAUDE.md §4 — this is a citation-trust product): expected citations are
**never authored from model knowledge.** They are *proposed* by the real retriever
(``scripts/build_golden_candidates.py``) and **confirmed by a human**.
``verification_status`` records how trustworthy each case is; only
``human_verified`` / ``lawyer_verified`` cases drive the headline metrics. The
machinery here builds + loads cases; a human populates the verified answers.

Pure standard library (CLAUDE.md §6): JSONL via ``json``, no external deps.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

# A case is a holding (a specific ruling), an issue (the legal question framed), or
# a refusal probe (must be declined — unsupported / out-of-jurisdiction).
CASE_TYPES = ("holding", "issue", "refusal")

# How trustworthy a case's expected citations are. Only the headline tiers below
# count toward reported headline metrics; the rest are seeds / directional signal.
VERIFICATION_STATUSES = (
    "unverified",            # a seed query; no confirmed citations yet (NOT scored)
    "machine_bootstrapped",  # retriever-proposed, not yet human-checked (directional)
    "human_verified",        # a human confirmed the citation is real + on point
    "lawyer_verified",       # a qualified lawyer confirmed it
)
#: Statuses trustworthy enough to headline (the rest are directional/seeds).
HEADLINE_STATUSES = ("human_verified", "lawyer_verified")


@dataclass
class GoldenCase:
    """One golden expectation. Expected citations come from the retriever + a human,
    NEVER from a model (see module docstring / CLAUDE.md §4)."""

    id: str
    question: str
    case_type: str  # one of CASE_TYPES
    #: Real INSC neutral citations present in the index, e.g. "2024INSC846".
    expected_doc_ids: list[str] = field(default_factory=list)
    #: Plain-English holding, human-authored from the cited source (NOT the model).
    expected_holding: str = ""
    #: Optional: the specific chunk ids a human confirmed support the holding.
    supporting_chunk_ids: list[str] = field(default_factory=list)
    verification_status: str = "unverified"
    #: The seed query this case was bootstrapped from (provenance).
    source_query: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.case_type not in CASE_TYPES:
            raise ValueError(
                f"case_type must be one of {CASE_TYPES}, got {self.case_type!r}"
            )
        if self.verification_status not in VERIFICATION_STATUSES:
            raise ValueError(
                f"verification_status must be one of {VERIFICATION_STATUSES}, "
                f"got {self.verification_status!r}"
            )

    @property
    def is_headline(self) -> bool:
        """True if this case is trustworthy enough to drive headline metrics."""
        return self.verification_status in HEADLINE_STATUSES

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenCase":
        """Build from a JSON object, ignoring unknown keys and filling defaults."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    """Load cases from a JSONL file. Blank lines and ``#``/``//`` comment lines are
    skipped (so the committed template can carry human-facing comments). A missing
    file returns an empty list (the golden set may not be populated yet)."""
    p = Path(path)
    if not p.exists():
        return []
    cases: list[GoldenCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        cases.append(GoldenCase.from_dict(json.loads(stripped)))
    return cases


def save_golden_cases(cases: Iterable[GoldenCase], path: str | Path) -> int:
    """Write cases as pure JSONL (no comments). Returns the count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def filter_by_status(
    cases: Iterable[GoldenCase], statuses: str | Iterable[str]
) -> list[GoldenCase]:
    """Return only the cases whose ``verification_status`` is in ``statuses``."""
    wanted = {statuses} if isinstance(statuses, str) else set(statuses)
    return [c for c in cases if c.verification_status in wanted]
