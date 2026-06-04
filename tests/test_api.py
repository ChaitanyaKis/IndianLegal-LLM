"""Offline integration tests for the FastAPI surface.

Hits the real endpoints via Starlette's TestClient using a STUB-pinned pipeline,
so it runs offline in CI (no GPU, no network). Skipped if fastapi/httpx absent.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None,
    reason="fastapi/httpx not installed (API integration test)",
)


def _client():
    from fastapi.testclient import TestClient

    from indianlegal_llm.app.api import create_app
    from indianlegal_llm.ingestion.stub import StubIngestor
    from indianlegal_llm.model.stub import StubLLM
    from indianlegal_llm.pipeline import build_pipeline

    pipeline = build_pipeline(ingestor=StubIngestor(), llm=StubLLM())
    return TestClient(create_app(pipeline=pipeline))


def test_health_reports_backends():
    resp = _client().get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0
    assert body["llm_backend"] == "StubLLM"


def test_answer_returns_structured_citations():
    resp = _client().post("/answer", json={"question": "Is privacy a fundamental right in India?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["is_grounded"] is True
    assert body["refusal_reason"] == ""
    assert body["citations"], "expected at least one citation"
    cite = body["citations"][0]
    # Fully structured citation fields.
    for field in (
        "chunk_id",
        "title",
        "neutral_citation",
        "reference",
        "pinpoint",
        "para_start",
        "para_end",
        "url",
    ):
        assert field in cite
    assert cite["title"] == "K. S. Puttaswamy v. Union of India"
    assert cite["url"].startswith("http")


def test_answer_refusal_carries_reason():
    resp = _client().post("/answer", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert body["refusal_reason"]  # a non-empty, structured reason
