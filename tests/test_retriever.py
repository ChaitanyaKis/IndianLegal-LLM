"""Tests for retriever selection + the cross-lingual e5 retriever plumbing (offline)."""

from __future__ import annotations

import importlib.util

import pytest

from indianlegal_llm.config import Settings
from indianlegal_llm.ingestion.stub import StubIngestor
from indianlegal_llm.model.stub import StubLLM
from indianlegal_llm.pipeline import build_pipeline
from indianlegal_llm.rag.embedding_retriever import (
    EmbeddingRetriever,
    MultilingualE5Embedder,
    _should_shard,
    cuda_device_count,
)


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


def test_should_shard_only_with_multi_gpu_and_enough_texts():
    # Multi-GPU data-parallel embedding engages only with >1 device AND a corpus
    # at/over the threshold AND a positive threshold.
    assert _should_shard(60_000, device_count=2, threshold=50_000) is True
    assert _should_shard(50_000, device_count=4, threshold=50_000) is True  # boundary
    assert _should_shard(60_000, device_count=1, threshold=50_000) is False  # 1 GPU
    assert _should_shard(10_000, device_count=2, threshold=50_000) is False  # too few
    assert _should_shard(60_000, device_count=2, threshold=0) is False       # disabled


def test_cuda_device_count_never_raises_and_is_non_negative():
    # Detection must degrade gracefully (0 with no torch/GPU, never an exception).
    n = cuda_device_count()
    assert isinstance(n, int) and n >= 0


def test_embedder_reads_multi_gpu_threshold_from_env(monkeypatch):
    monkeypatch.setenv("E5_MULTI_GPU_MIN_CHUNKS", "12345")
    assert MultilingualE5Embedder().multi_gpu_min_texts == 12345
    monkeypatch.setenv("E5_MULTI_GPU_MIN_CHUNKS", "garbage")
    assert MultilingualE5Embedder().multi_gpu_min_texts == 50_000  # safe default
    # An explicit constructor arg overrides the env.
    assert MultilingualE5Embedder(multi_gpu_min_texts=7).multi_gpu_min_texts == 7
