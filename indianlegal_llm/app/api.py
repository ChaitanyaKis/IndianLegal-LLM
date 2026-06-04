"""FastAPI surface (optional dependency, import-guarded).

Importing this module never fails: if FastAPI is not installed, ``app`` is None
and ``create_app()`` raises a clear error. Install with::

    pip install -e .[api]

Run with::

    uvicorn indianlegal_llm.app.api:app --reload
"""

from __future__ import annotations

import functools

from ..pipeline import build_pipeline
from ..schemas import Answer


@functools.lru_cache(maxsize=1)
def _get_pipeline():
    """Build the pipeline once, on first request — not at import time."""
    return build_pipeline()

try:  # optional dependency — guarded so the package imports with stdlib only
    from fastapi import FastAPI
    from pydantic import BaseModel

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - exercised only without fastapi installed
    _HAS_FASTAPI = False


def _serialize(answer: Answer) -> dict:
    return {
        "question": answer.question,
        "text": answer.text,
        "refused": answer.refused,
        "is_grounded": answer.is_grounded,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "court": c.court,
                "url": c.url,
            }
            for c in answer.citations
        ],
    }


def create_app():
    """Build the FastAPI app. Raises if FastAPI is not installed."""
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install the API extra: pip install -e .[api]"
        )

    fastapi_app = FastAPI(
        title="IndianLegal-LLM",
        description="Citation-grounded LLM framework for Indian law.",
        version="0.1.0",
    )

    class Query(BaseModel):
        question: str

    @fastapi_app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks_indexed": _get_pipeline().num_chunks}

    @fastapi_app.post("/answer")
    def answer(query: Query) -> dict:
        return _serialize(_get_pipeline().answer(query.question))

    return fastapi_app


# Module-level ASGI app for `uvicorn ...:app` — only when FastAPI is available.
app = create_app() if _HAS_FASTAPI else None
