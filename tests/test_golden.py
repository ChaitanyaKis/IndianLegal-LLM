"""Offline tests for the golden eval set: schema/loader round-trip, status filter,
and the scorer (a retrieval hit, a miss, a correct refusal, a wrong refusal).

No GPU / torch / network: the scorer is exercised with a fake pipeline built from
the real schemas (Chunk/RetrievedChunk/Citation/Answer), so attribute access and
the citation guard match production exactly.
"""

from __future__ import annotations

import types

import pytest

from indianlegal_llm.evaluation.golden import (
    GoldenCase,
    filter_by_status,
    load_golden_cases,
    save_golden_cases,
)
from indianlegal_llm.evaluation.quality import score_golden_cases
from indianlegal_llm.schemas import Answer, Chunk, Citation, RetrievedChunk


# --------------------------------------------------------------------------- #
# Schema + loader
# --------------------------------------------------------------------------- #
def test_goldencase_jsonl_round_trip(tmp_path):
    cases = [
        GoldenCase(
            id="c1", question="q1?", case_type="holding",
            expected_doc_ids=["2024INSC846"], expected_holding="the court held X",
            supporting_chunk_ids=["sc-2024INSC846::5"],
            verification_status="human_verified", source_query="q1?", notes="n",
        ),
        GoldenCase(id="c2", question="q2?", case_type="refusal"),
    ]
    path = tmp_path / "golden.jsonl"
    assert save_golden_cases(cases, path) == 2
    loaded = load_golden_cases(path)
    assert [c.to_dict() for c in loaded] == [c.to_dict() for c in cases]


def test_loader_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "# a comment\n"
        "\n"
        '{"id": "x", "question": "q?", "case_type": "issue", '
        '"verification_status": "unverified"}\n'
        "// another comment\n",
        encoding="utf-8",
    )
    cases = load_golden_cases(path)
    assert len(cases) == 1 and cases[0].id == "x"


def test_loader_missing_file_is_empty(tmp_path):
    assert load_golden_cases(tmp_path / "nope.jsonl") == []


def test_from_dict_ignores_unknown_keys_and_fills_defaults():
    case = GoldenCase.from_dict(
        {"id": "x", "question": "q?", "case_type": "issue", "bogus": 1}
    )
    assert case.expected_doc_ids == [] and case.verification_status == "unverified"


def test_invalid_case_type_and_status_rejected():
    with pytest.raises(ValueError):
        GoldenCase(id="x", question="q?", case_type="opinion")
    with pytest.raises(ValueError):
        GoldenCase(id="x", question="q?", case_type="issue", verification_status="meh")


def test_filter_by_status_and_is_headline():
    cases = [
        GoldenCase(id="a", question="q", case_type="issue", verification_status="unverified"),
        GoldenCase(id="b", question="q", case_type="issue", verification_status="human_verified"),
        GoldenCase(id="c", question="q", case_type="issue", verification_status="lawyer_verified"),
        GoldenCase(id="d", question="q", case_type="issue", verification_status="machine_bootstrapped"),
    ]
    headline = filter_by_status(cases, ("human_verified", "lawyer_verified"))
    assert {c.id for c in headline} == {"b", "c"}
    assert all(c.is_headline for c in headline)
    assert not cases[0].is_headline and not cases[3].is_headline


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #
def _chunk(doc_id, nc, idx=0, text="placeholder text"):
    return Chunk(
        chunk_id=f"{doc_id}::{idx}", doc_id=doc_id, text=text,
        title="X v. Y", court="Supreme Court of India", url="", license="gov",
        metadata={"nc_display": nc, "para_start": 12, "para_end": 12},
    )


def _fake_pipeline(retrieved_by_q: dict, answers_by_q: dict):
    retriever = types.SimpleNamespace(retrieve=lambda q, k: retrieved_by_q.get(q, []))
    answerer = types.SimpleNamespace(
        retriever=retriever, llm=types.SimpleNamespace(model_id="stub")
    )
    pipe = types.SimpleNamespace(
        answerer=answerer,
        settings=types.SimpleNamespace(top_k=3),
        source="local-sc",
        answer=lambda q: answers_by_q[q],
    )
    # llm_backend is a property on the real Pipeline; a plain attr is fine here.
    pipe.llm_backend = "StubLLM"
    return pipe


def test_score_golden_hit_miss_correct_and_wrong_refusal():
    # --- a HIT: holding case, right doc retrieved + cited, holding grounded -----
    hit_text = "the reassessment was without jurisdiction and is quashed"
    hit_chunk = _chunk("sc-2024INSC846", "2024INSC846", text=hit_text)
    hit_q = "holding question?"
    hit_answer = Answer(
        question=hit_q, text="...",
        citations=[Citation(
            chunk_id=hit_chunk.chunk_id, doc_id="sc-2024INSC846", title="X v. Y",
            court="SC", url="", neutral_citation="2024INSC846", para_start=12,
        )],
        refused=False,
    )

    # --- a MISS: expected doc not retrieved, system refused --------------------
    miss_q = "issue question?"
    miss_chunk = _chunk("sc-2024INSC111", "2024INSC111")
    miss_answer = Answer(question=miss_q, text="", citations=[], refused=True)

    # --- a CORRECT refusal: refusal probe, system refused ----------------------
    refuse_ok_q = "foreign law question?"
    refuse_ok_answer = Answer(question=refuse_ok_q, text="", citations=[], refused=True)

    # --- a WRONG refusal: refusal probe, system answered (fabricated citation) --
    refuse_bad_q = "unsupported question?"
    bad_chunk = _chunk("sc-2024INSC222", "2024INSC222")
    refuse_bad_answer = Answer(
        question=refuse_bad_q, text="...",
        citations=[Citation(
            chunk_id=bad_chunk.chunk_id, doc_id="sc-2024INSC222", title="A v. B",
            court="SC", url="", neutral_citation="2024INSC222",
        )],
        refused=False,
    )

    pipe = _fake_pipeline(
        retrieved_by_q={
            hit_q: [RetrievedChunk(chunk=hit_chunk, score=0.86)],
            miss_q: [RetrievedChunk(chunk=miss_chunk, score=0.82)],
            refuse_ok_q: [],
            refuse_bad_q: [RetrievedChunk(chunk=bad_chunk, score=0.81)],
        },
        answers_by_q={
            hit_q: hit_answer, miss_q: miss_answer,
            refuse_ok_q: refuse_ok_answer, refuse_bad_q: refuse_bad_answer,
        },
    )

    cases = [
        GoldenCase(id="hit", question=hit_q, case_type="holding",
                   expected_doc_ids=["2024INSC846"], expected_holding=hit_text,
                   verification_status="human_verified"),
        GoldenCase(id="miss", question=miss_q, case_type="issue",
                   expected_doc_ids=["2099INSC999"], verification_status="human_verified"),
        GoldenCase(id="refuse_ok", question=refuse_ok_q, case_type="refusal",
                   verification_status="human_verified"),
        GoldenCase(id="refuse_bad", question=refuse_bad_q, case_type="refusal",
                   verification_status="human_verified"),
        # a seed that must be COUNTED but never scored
        GoldenCase(id="seed", question="seed?", case_type="issue",
                   verification_status="unverified"),
    ]

    report = score_golden_cases(pipe, cases)

    assert report["status_counts"] == {"human_verified": 4, "unverified": 1}
    h = report["headline"]
    assert h["n_cases"] == 4 and h["n_holding_issue"] == 2 and h["n_refusal"] == 2
    assert h["retrieval_hit_rate"] == 0.5   # hit True, miss False
    assert h["citation_accuracy"] == 0.5    # hit cited the right doc; miss refused
    assert h["proposition_grounding"] == 1.0  # only the hit had an expected_holding
    assert h["refusal_correctness"] == 0.5  # one correct refusal, one fabricated

    per = {r["id"]: r for r in report["per_case"]}
    assert "seed" not in per  # unverified seed never scored
    assert per["hit"]["retrieval_hit"] is True and per["hit"]["citation_correct"] is True
    assert per["miss"]["retrieval_hit"] is False
    assert per["refuse_ok"]["refusal_correct"] is True
    assert per["refuse_bad"]["refusal_correct"] is False
    assert per["refuse_bad"]["fabricated_citation"] is True


def test_score_golden_machine_bootstrapped_is_directional_not_headline():
    q = "issue question?"
    chunk = _chunk("sc-2024INSC846", "2024INSC846")
    pipe = _fake_pipeline(
        {q: [RetrievedChunk(chunk=chunk, score=0.84)]},
        {q: Answer(question=q, text="", citations=[], refused=True)},
    )
    cases = [
        GoldenCase(id="mb", question=q, case_type="issue",
                   expected_doc_ids=["2024INSC846"],
                   verification_status="machine_bootstrapped"),
    ]
    report = score_golden_cases(pipe, cases)
    assert report["headline"]["n_cases"] == 0      # nothing headlines
    assert report["directional"]["n_cases"] == 1   # reported as directional
    assert report["directional"]["retrieval_hit_rate"] == 1.0
