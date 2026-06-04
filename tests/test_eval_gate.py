"""Tests for the two-tier evaluation: golden set, deterministic gate, quality eval."""

from __future__ import annotations

import json
from pathlib import Path

from indianlegal_llm.evaluation import harness, quality
from indianlegal_llm.evaluation.cases import (
    CASES,
    deterministic_cases,
    load_golden_set,
    quality_cases,
)
from indianlegal_llm.evaluation.schema import TIER_DETERMINISTIC, TIER_QUALITY
from indianlegal_llm.ingestion.stub import StubIngestor
from indianlegal_llm.model.stub import StubLLM
from indianlegal_llm.pipeline import build_pipeline

_BASELINE = Path(harness.__file__).with_name("baseline_metrics.json")


def _stub_pipeline():
    return build_pipeline(ingestor=StubIngestor(), llm=StubLLM())


# --------------------------------------------------------------------------- #
# Golden set
# --------------------------------------------------------------------------- #
def test_golden_set_loads_and_partitions_by_tier():
    everything = load_golden_set()
    assert everything
    det = deterministic_cases()
    qual = quality_cases()
    assert det == CASES
    assert all(TIER_DETERMINISTIC in c.tiers for c in det)
    assert all(TIER_QUALITY in c.tiers for c in qual)
    # The seed deterministic tier: 5 English cases + 2 Indic out-of-corpus refusals.
    assert {c.case_id for c in det} == {
        "privacy",
        "basic-structure",
        "capital-of-france",
        "european-union-structure",
        "us-bill-of-rights",
        "hindi-out-of-corpus",
        "telugu-out-of-corpus",
    }


def test_todo_placeholders_are_inert():
    everything = load_golden_set()
    todos = [c for c in everything if c.is_pending]
    assert todos, "expected lawyer TODO slots in the golden set"
    pending_ids = {c.case_id for c in todos}
    # TODO placeholders never reach either runner.
    assert pending_ids.isdisjoint({c.case_id for c in deterministic_cases()})
    assert pending_ids.isdisjoint({c.case_id for c in quality_cases()})


def test_quality_cases_carry_lawyer_fields():
    for case in quality_cases():
        assert case.expected_authority  # neutral citation / case for verification


# --------------------------------------------------------------------------- #
# Deterministic gate
# --------------------------------------------------------------------------- #
def test_gate_is_green_with_full_metrics():
    metrics = harness.run()
    assert metrics["citation_accuracy"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["retrieval_hit_rate"] == 1.0
    assert metrics["hallucinations"] == 0
    assert metrics["guard_invariants_ok"] is True
    assert metrics["green"] is True


def test_guard_self_check_passes():
    ok, failures = harness.guard_self_check()
    assert ok, f"guard invariants broke: {failures}"
    assert failures == []


def test_baseline_metrics_match_current():
    """The committed baseline must equal the current scalars (keeps delta honest)."""
    current = harness.metrics_scalars(harness.run())
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert current == baseline


def test_gate_main_returns_zero_and_can_emit_artifacts(tmp_path):
    assert harness.main() == 0  # no args -> no file writes
    json_path = tmp_path / "metrics.json"
    summary_path = tmp_path / "summary.md"
    assert harness.main(["--json", str(json_path), "--summary", str(summary_path)]) == 0
    metrics = json.loads(json_path.read_text(encoding="utf-8"))
    assert metrics["green"] is True
    summary = summary_path.read_text(encoding="utf-8")
    assert "deterministic eval gate" in summary
    assert "GATE: GREEN" in summary


# --------------------------------------------------------------------------- #
# Quality eval (NON-blocking) — runs offline via the stub fallback
# --------------------------------------------------------------------------- #
def test_quality_eval_runs_offline_and_is_nonblocking():
    report = quality.run(pipeline=_stub_pipeline())
    assert report["num_cases"] == len(quality_cases())
    assert report["is_real_run"] is False  # stub fallback offline
    assert report["citation_accuracy"] is not None
    assert "legalbench" in report["task_metrics"]
    # main() is non-blocking: always exit 0
    assert quality.main([]) == 0
