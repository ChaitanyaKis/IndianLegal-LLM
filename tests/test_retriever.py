"""Tests for retriever selection + the cross-lingual e5 retriever plumbing (offline)."""

from __future__ import annotations

import importlib.util

import pytest

from indianlegal_llm.config import Settings
from indianlegal_llm.ingestion.stub import StubIngestor
from indianlegal_llm.model.stub import StubLLM
from indianlegal_llm.pipeline import build_pipeline
from indianlegal_llm.rag.embedding_retriever import EmbeddingRetriever, MultilingualE5Embedder


def test_embedding_retriever_constructs_without_loading():
    retriever = EmbeddingRetriever()
    assert isinstance(retriever.embedder, MultilingualE5Embedder)
    assert retriever._matrix is None
    assert retriever.embedder._model is None  # sentence-transformers not loaded


def test_default_vector_backend_is_lexical():
    pipe = build_pipeline(Settings(), ingestor=StubIngestor(), llm=StubLLM())
    assert type(pipe.answerer.retriever).__name__ == "InMemoryRetriever"


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="sentence-transformers installed; the offline fallback isn't exercised",
)
def test_vector_backend_e5_falls_back_to_lexical_when_unavailable(capsys):
    pipe = build_pipeline(
        Settings(vector_backend="e5"), ingestor=StubIngestor(), llm=StubLLM()
    )
    assert type(pipe.answerer.retriever).__name__ == "InMemoryRetriever"
    assert "lexical InMemoryRetriever" in capsys.readouterr().err
