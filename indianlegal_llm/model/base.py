"""LLM interface.

A model receives a system prompt and a user prompt and returns raw text. It is
deliberately unaware of citations and refusals — those are enforced downstream by
the Answerer. This keeps the model swappable: any base model (Apache-2.0/MIT per
CLAUDE.md §2), with or without an MIT adapter, can sit behind this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract text-in / text-out language model."""

    #: Identifier of the underlying model (for logs / provenance).
    model_id: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the model's completion for the given system and user prompts."""
        raise NotImplementedError
