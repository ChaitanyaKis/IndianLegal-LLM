"""Tests for real-model serving.

The real-model path is gated behind GPU + transformers availability (skipped when
absent). The offline tests assert the registry and the stub fallback. Eval stays
on the stub (see test_pipeline / harness), so it is deterministic and GPU-free.
"""

from __future__ import annotations

import importlib.util

import pytest

from indianlegal_llm.config import Settings
from indianlegal_llm.ingestion.stub import StubIngestor
from indianlegal_llm.model.registry import get_llm
from indianlegal_llm.model.stub import StubLLM
from indianlegal_llm.model.transformers_llm import TransformersLLM
from indianlegal_llm.pipeline import build_pipeline


def _real_model_available() -> bool:
    """True only when transformers + a CUDA GPU are present (else skip)."""
    if importlib.util.find_spec("torch") is None:
        return False
    if importlib.util.find_spec("transformers") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch import edge cases
        return False


# --------------------------------------------------------------------------- #
# Offline: registry + construction (no torch import, no weight download)
# --------------------------------------------------------------------------- #
def test_get_llm_resolves_backends_without_loading():
    assert isinstance(get_llm("stub"), StubLLM)
    llm = get_llm("transformers", base_model="microsoft/phi-4")
    assert isinstance(llm, TransformersLLM)
    assert llm.model_id == "microsoft/phi-4"
    assert llm._model is None  # constructing must NOT import torch or load weights
    with pytest.raises(ValueError):
        get_llm("gpt-4o")  # unknown backend


def test_transformers_llm_defaults_to_deterministic_generation():
    llm = get_llm("transformers")
    assert llm.temperature == 0.0  # greedy decoding for legal determinism
    assert llm.max_new_tokens > 0


# --------------------------------------------------------------------------- #
# Offline: graceful fallback keeps the skeleton running (CLAUDE.md §6)
# --------------------------------------------------------------------------- #
def test_build_pipeline_falls_back_to_stub_when_real_llm_unavailable(monkeypatch, capsys):
    from indianlegal_llm import pipeline as pl

    def _boom(*args, **kwargs):
        raise ImportError("torch not installed (simulated)")

    monkeypatch.setattr(pl, "get_llm", _boom)
    pipe = pl.build_pipeline(Settings(llm="transformers"), ingestor=StubIngestor())

    assert pipe.llm_backend == "StubLLM"  # degraded to the offline stub
    assert pipe.answer("Is privacy a fundamental right in India?").is_grounded
    assert "falling back to StubLLM" in capsys.readouterr().err


def test_no_gpu_raises_before_downloading_weights(monkeypatch):
    """ensure_loaded must refuse on a CPU/laptop BEFORE any weight download."""
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        pytest.skip("torch not installed")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        TransformersLLM(model_id="microsoft/phi-4").ensure_loaded()


def test_explicit_llm_is_never_overridden():
    pipe = build_pipeline(
        Settings(llm="transformers"), ingestor=StubIngestor(), llm=StubLLM()
    )
    assert pipe.llm_backend == "StubLLM"


# --------------------------------------------------------------------------- #
# GPU smoke test: real Phi-4 answer must pass the citation guard
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _real_model_available(),
    reason="no CUDA GPU / transformers; real-model path skipped",
)
def test_phi4_smoke_answer_passes_guard():  # pragma: no cover - runs only on a GPU
    pipe = build_pipeline(
        Settings(llm="transformers", base_model="microsoft/phi-4"),
        ingestor=StubIngestor(),
    )
    assert pipe.llm_backend == "TransformersLLM"  # the real model actually loaded
    answer = pipe.answer("Is privacy a fundamental right in India?")
    # Guard invariant either way: a non-refused answer is grounded with valid,
    # rendered citations; a refusal carries none. No ungrounded claim escapes.
    if answer.refused:
        assert answer.citations == []
    else:
        assert answer.is_grounded and answer.citations
        assert all(c.reference for c in answer.citations)
