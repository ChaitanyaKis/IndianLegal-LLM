"""Retrieval-augmented generation layer.

Interfaces: :class:`~indianlegal_llm.rag.base.BaseEmbedder`,
:class:`~indianlegal_llm.rag.base.BaseRetriever`.
Skeleton stubs: :class:`~indianlegal_llm.rag.embedder.StubEmbedder`,
:class:`~indianlegal_llm.rag.retriever.InMemoryRetriever`.

The :class:`~indianlegal_llm.rag.answerer.Answerer` enforces the trust property
(CLAUDE.md §4): no answer is returned unless it cites a retrieved source.
"""

from .answerer import Answerer
from .base import BaseEmbedder, BaseRetriever
from .citation import (
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_cited_ids,
    to_citation,
)
from .embedder import StubEmbedder
from .retriever import InMemoryRetriever

__all__ = [
    "BaseEmbedder",
    "BaseRetriever",
    "StubEmbedder",
    "InMemoryRetriever",
    "Answerer",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "extract_cited_ids",
    "to_citation",
]
