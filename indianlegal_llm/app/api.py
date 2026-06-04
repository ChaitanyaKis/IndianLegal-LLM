"""FastAPI surface (optional dependency, import-guarded).

Importing this module never fails: if FastAPI is not installed, ``app`` is None
and ``create_app()`` raises a clear error. Install with::

    pip install -e .[api]

Run with::

    uvicorn indianlegal_llm.app.api:app --reload

Endpoints:
- ``GET /health``  -> liveness + which backends are serving.
- ``POST /answer`` -> {question, text, refused, refusal_reason, is_grounded,
  citations[...]} where each citation is fully structured (chunk_id, title,
  neutral_citation, reference, pinpoint, para_start/para_end, url). The pipeline
  applies the citation guard, so a non-refused answer is always grounded.
"""

from __future__ import annotations

import functools

from ..pipeline import Pipeline, build_pipeline
from ..schemas import Answer

try:  # optional dependency — guarded so the package imports with stdlib only
    from fastapi import FastAPI
    from pydantic import BaseModel

    _HAS_FASTAPI = True

    class AnswerRequest(BaseModel):
        """POST /answer body. Defined at module level so FastAPI can resolve the
        annotation under ``from __future__ import annotations``."""

        question: str

except ImportError:  # pragma: no cover - exercised only without fastapi installed
    _HAS_FASTAPI = False


@functools.lru_cache(maxsize=1)
def _get_pipeline() -> Pipeline:
    """Build the pipeline once, on first request — not at import time."""
    return build_pipeline()


def _serialize(answer: Answer) -> dict:
    return {
        "question": answer.question,
        "text": answer.text,
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "is_grounded": answer.is_grounded,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "court": c.court,
                "neutral_citation": c.neutral_citation,
                "reference": c.reference,
                "pinpoint": c.pinpoint,
                "para_start": c.para_start,
                "para_end": c.para_end,
                "url": c.url,
            }
            for c in answer.citations
        ],
    }


def create_app(pipeline: Pipeline | None = None):
    """Build the FastAPI app. Raises if FastAPI is not installed.

    ``pipeline`` may be supplied (e.g. a stub-pinned pipeline) for offline tests;
    otherwise it is built lazily on first request.
    """
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install the API extra: pip install -e .[api]"
        )

    fastapi_app = FastAPI(
        title="IndianLegal-LLM",
        description="Citation-grounded question answering for Indian law.",
        version="0.1.0",
    )

    def get_pipeline() -> Pipeline:
        return pipeline if pipeline is not None else _get_pipeline()

    @fastapi_app.get("/health")
    def health() -> dict:
        pipe = get_pipeline()
        return {
            "status": "ok",
            "chunks_indexed": pipe.num_chunks,
            "ingestor": pipe.source,
            "llm_backend": pipe.llm_backend,
        }

    @fastapi_app.post("/answer")
    def answer(request: AnswerRequest) -> dict:
        return _serialize(get_pipeline().answer(request.question))

    return fastapi_app


# Module-level ASGI app for `uvicorn ...:app` — only when FastAPI is available.
app = create_app() if _HAS_FASTAPI else None
