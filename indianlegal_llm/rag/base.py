"""RAG interfaces: embedding and retrieval.

The skeleton's :class:`~indianlegal_llm.rag.retriever.InMemoryRetriever` is purely
lexical (token overlap) and does not require an embedder. :class:`BaseEmbedder`
exists so the embedding workstream can later add a vector retriever behind the
same :class:`BaseRetriever` contract without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Chunk, RetrievedChunk


class BaseEmbedder(ABC):
    """Maps text into dense vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string."""
        raise NotImplementedError


class BaseRetriever(ABC):
    """Indexes chunks and returns the most relevant ones for a query."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index. May be called more than once."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks ranked by relevance (highest first).

        Implementations MUST NOT return irrelevant chunks: a chunk with no
        relevance to the query must be omitted. This is what lets the Answerer
        refuse on out-of-domain questions (CLAUDE.md §4).
        """
        raise NotImplementedError
