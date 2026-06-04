"""Stub embedder: a deterministic, dependency-free hashing bag-of-words vector.

This is a placeholder for the embedding workstream. It is NOT used by the
skeleton's lexical :class:`InMemoryRetriever`; it exists so a future vector
retriever has a concrete :class:`BaseEmbedder` to start from. Being deterministic
(no randomness, no model download) it keeps the skeleton reproducible.
"""

from __future__ import annotations

import hashlib

from .base import BaseEmbedder


class StubEmbedder(BaseEmbedder):
    """Hashes tokens into a fixed-size vector. Deterministic; no dependencies."""

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = digest[0] % self.dim
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]
