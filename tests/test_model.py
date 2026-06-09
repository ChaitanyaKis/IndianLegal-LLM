"""Tests for real-model serving.

The real-model path is gated behind GPU + transformers availability (skipped when
absent). The offline tests assert the registry and the stub fallback. Eval stays
on the stub (see test_pipeline / harness), so it is deterministic and GPU-free.
"""

from __future__ import annotations

import importlib.util
import os

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


def test_quant_4bit_config_has_true_4bit_fields():
    """The 4-bit knobs must be present + correct so a ~14B model fits a T4 (~9 GB,
    not ~16+). Asserted offline on the data, without importing transformers/torch."""
    from indianlegal_llm.model.transformers_llm import QUANT_4BIT

    assert QUANT_4BIT["load_in_4bit"] is True
    assert QUANT_4BIT["bnb_4bit_quant_type"] == "nf4"
    assert QUANT_4BIT["bnb_4bit_use_double_quant"] is True
    # A dtype NAME (resolved to torch.bfloat16 at load), so the field set is
    # checkable without torch; compute dtype must NOT be a full-precision float32.
    assert QUANT_4BIT["bnb_4bit_compute_dtype"] == "bfloat16"


def test_small_eval_model_is_on_the_license_allowlist():
    """The default eval model must be license-clean (CLAUDE.md §2) in both the
    config allowlist and the serving allowlist."""
    from indianlegal_llm.config import LICENSE_CLEAN_BASE_MODELS
    from indianlegal_llm.model.transformers_llm import _LICENSE_CLEAN

    assert LICENSE_CLEAN_BASE_MODELS.get("microsoft/Phi-3.5-mini-instruct") == "MIT"
    assert "microsoft/Phi-3.5-mini-instruct" in _LICENSE_CLEAN


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
# Remote LLM backend (HTTP; no GPU/weights)
# --------------------------------------------------------------------------- #
def test_get_llm_resolves_remote_backend():
    from indianlegal_llm.model.remote_llm import RemoteLLM

    assert isinstance(get_llm("remote"), RemoteLLM)


def test_remote_llm_requires_endpoint(monkeypatch):
    from indianlegal_llm.model.remote_llm import RemoteLLM

    monkeypatch.delenv("REMOTE_LLM_URL", raising=False)
    with pytest.raises(RuntimeError, match="REMOTE_LLM_URL"):
        RemoteLLM().ensure_loaded()
    # Configured -> validates without a network call.
    RemoteLLM(url="https://example.com/v1/chat/completions").ensure_loaded()


def test_build_pipeline_remote_unconfigured_falls_back(monkeypatch, capsys):
    monkeypatch.delenv("REMOTE_LLM_URL", raising=False)
    pipe = build_pipeline(Settings(llm="remote"), ingestor=StubIngestor())
    assert pipe.llm_backend == "StubLLM"
    assert "falling back to StubLLM" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Offline: LoRA adapter plumbing (the fine-tuned variant)
# --------------------------------------------------------------------------- #
def test_transformers_llm_accepts_adapter_without_loading():
    llm = get_llm(
        "transformers",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        adapter="/path/to/adapter",
    )
    assert isinstance(llm, TransformersLLM)
    assert llm.adapter == "/path/to/adapter"
    assert llm._model is None  # no base or adapter load at construction


def test_adapter_threads_through_build_pipeline(monkeypatch):
    from indianlegal_llm import pipeline as pl

    captured: dict = {}

    def _spy(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        raise ImportError("simulated: do not actually load weights")

    monkeypatch.setattr(pl, "get_llm", _spy)
    pl.build_pipeline(
        Settings(
            llm="transformers",
            base_model="Qwen/Qwen3-4B-Instruct-2507",
            adapter="/path/to/adapter",
        ),
        ingestor=StubIngestor(),
    )
    assert captured["name"] == "transformers"
    assert captured["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert captured["adapter"] == "/path/to/adapter"


def test_adapter_load_is_also_gpu_gated(monkeypatch):
    """Even with an adapter configured, no GPU -> raise before any download."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    llm = TransformersLLM(model_id="Qwen/Qwen3-4B-Instruct-2507", adapter="/a")
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        llm.ensure_loaded()


# --------------------------------------------------------------------------- #
# GPU smoke: Qwen3-4B base + LoRA adapter must pass the citation guard
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (_real_model_available() and os.getenv("TEST_ADAPTER")),
    reason="no CUDA GPU or TEST_ADAPTER unset; adapter smoke skipped",
)
def test_qwen3_with_adapter_smoke_passes_guard():  # pragma: no cover - GPU only
    pipe = build_pipeline(
        Settings(
            llm="transformers",
            base_model="Qwen/Qwen3-4B-Instruct-2507",
            adapter=os.environ["TEST_ADAPTER"],
        ),
        ingestor=StubIngestor(),
    )
    assert pipe.llm_backend == "TransformersLLM"  # base + adapter loaded
    answer = pipe.answer("Is privacy a fundamental right in India?")
    if answer.refused:
        assert answer.citations == []
    else:
        assert answer.is_grounded and answer.citations
        assert all(c.reference for c in answer.citations)


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
