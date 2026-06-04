"""Model layer: the LLM that turns a grounded prompt into a (cited) answer.

Interface: :class:`~indianlegal_llm.model.base.BaseLLM`.
Offline stub: :class:`~indianlegal_llm.model.stub.StubLLM`.
Real backend: :class:`~indianlegal_llm.model.transformers_llm.TransformersLLM`
(Phi-4 4-bit, served on a CUDA GPU; selected via the LLM flag / BASE_MODEL).

Use :func:`get_llm` to construct one by name; the real backend and its heavy
dependencies are imported lazily, so importing this package needs only the
standard library. The base model must be Apache-2.0 or MIT (CLAUDE.md §2).
"""

from .base import BaseLLM
from .registry import LLMS, get_llm
from .stub import StubLLM
from .transformers_llm import TransformersLLM

__all__ = ["BaseLLM", "StubLLM", "TransformersLLM", "get_llm", "LLMS"]
