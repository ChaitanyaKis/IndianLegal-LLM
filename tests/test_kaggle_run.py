"""Offline tests for the Kaggle runner's eval-LLM resolution + loud fallback chain.

The runner lives in scripts/ (not an installed package), so it is loaded via
importlib. No torch / GPU / sentence-transformers / S3 / network is touched here —
the fallback chain is exercised with an injected loader.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "kaggle_run.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("kaggle_run_under_test", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kr = _load_runner()


class _FakeLLM:
    """Stand-in for a successfully-loaded real LLM (NOT a StubLLM)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


def test_default_eval_model_is_the_small_mit_model():
    assert kr._DEFAULT_EVAL_LLM_MODEL == "microsoft/Phi-3.5-mini-instruct"


def test_resolve_llm_model_precedence():
    # flag > env LLM_MODEL > small default
    assert kr._resolve_llm_model("microsoft/phi-4", "x/y") == "microsoft/phi-4"
    assert kr._resolve_llm_model(None, "x/y") == "x/y"
    assert kr._resolve_llm_model("   ", None) == kr._DEFAULT_EVAL_LLM_MODEL
    assert kr._resolve_llm_model(None, "") == kr._DEFAULT_EVAL_LLM_MODEL


def test_load_real_llm_falls_back_to_small_model_then_succeeds():
    primary, fallback = "microsoft/phi-4", kr._DEFAULT_EVAL_LLM_MODEL
    tried: list[str] = []

    def loader(model_id, adapter):
        tried.append(model_id)
        if model_id == primary:
            raise RuntimeError("CUDA out of memory (simulated)")
        return _FakeLLM(model_id)

    llm = kr._load_real_llm(primary, fallback, loader=loader)
    assert isinstance(llm, _FakeLLM) and llm.model_id == fallback
    assert tried == [primary, fallback]  # tried primary, then fell back to small


def test_load_real_llm_raises_when_all_fail_never_returns_stub():
    def loader(model_id, adapter):
        raise RuntimeError("no GPU (simulated)")

    with pytest.raises(RuntimeError, match="could not load a real eval LLM"):
        kr._load_real_llm("microsoft/phi-4", kr._DEFAULT_EVAL_LLM_MODEL, loader=loader)


def test_load_real_llm_no_pointless_retry_when_primary_is_the_fallback():
    tried: list[str] = []

    def loader(model_id, adapter):
        tried.append(model_id)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        kr._load_real_llm(
            kr._DEFAULT_EVAL_LLM_MODEL, kr._DEFAULT_EVAL_LLM_MODEL, loader=loader
        )
    assert tried == [kr._DEFAULT_EVAL_LLM_MODEL]  # identical fallback not retried
