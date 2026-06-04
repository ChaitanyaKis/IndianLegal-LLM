"""Map an LLM backend name to a configured BaseLLM.

The real backend's heavy imports are deferred into the branch (and further into
``TransformersLLM.ensure_loaded``), so importing this module needs only the
standard library.
"""

from __future__ import annotations

from .base import BaseLLM
from .stub import StubLLM

LLMS = ("stub", "transformers")


def get_llm(name: str, base_model: str = "microsoft/phi-4", **kwargs) -> BaseLLM:
    """Return an LLM for ``name``. Raises ValueError on an unknown backend.

    A missing optional dependency surfaces later, from ``ensure_loaded`` /
    ``generate`` on the returned object — not from importing this module.
    """
    key = (name or "stub").strip().lower()
    if key == "stub":
        return StubLLM()
    if key in ("transformers", "phi-4", "phi4", "hf"):
        from .transformers_llm import TransformersLLM

        return TransformersLLM(model_id=base_model, **kwargs)
    raise ValueError(f"unknown LLM backend {name!r}; choose from {', '.join(LLMS)}")
