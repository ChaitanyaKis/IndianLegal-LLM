"""Model layer: the LLM that turns a grounded prompt into a (cited) answer.

Interface: :class:`~indianlegal_llm.model.base.BaseLLM`.
Skeleton stub: :class:`~indianlegal_llm.model.stub.StubLLM`.

Real implementations wrap a base model that is Apache-2.0 or MIT (CLAUDE.md §2),
optionally with an MIT-licensed fine-tuned adapter.
"""

from .base import BaseLLM
from .stub import StubLLM

__all__ = ["BaseLLM", "StubLLM"]
