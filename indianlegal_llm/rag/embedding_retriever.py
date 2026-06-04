"""Cross-lingual embedding retriever (multilingual-e5).

This is the real, cross-lingual ``BaseRetriever``: an Indian-language query
retrieves over the English corpus because ``intfloat/multilingual-e5`` (MIT) maps
both into a shared multilingual space. Selected via ``VECTOR_BACKEND=e5``;
``build_pipeline()`` falls back to the lexical :class:`InMemoryRetriever` when
``sentence-transformers`` is unavailable, so offline/CI stay on the deterministic
lexical path (the gate is never embedding-dependent).

A minimum cosine score keeps the refusal property: a query with no sufficiently
similar chunk retrieves nothing, so the Answerer refuses (CLAUDE.md §4). Heavy
deps (sentence-transformers, numpy) are imported lazily.
"""

from __future__ import annotations

import os

from ..schemas import Chunk, RetrievedChunk
from .base import BaseEmbedder, BaseRetriever

# intfloat/multilingual-e5-* are MIT-licensed and cover ~100 languages incl. the
# major Indian scripts. -small is CPU-friendly; override via EMBEDDING_MODEL.
_DEFAULT_E5_MODEL = "intfloat/multilingual-e5-small"


class MultilingualE5Embedder(BaseEmbedder):
    """Sentence-Transformers wrapper for multilingual-e5 (e5 needs role prefixes)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", _DEFAULT_E5_MODEL)
        self._model = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "sentence-transformers is required for the e5 retriever. "
                "Install the rag extra: pip install -e .[rag]"
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def _encode(self, texts: list[str], prefix: str):
        self.ensure_loaded()
        return self._model.encode(
            [f"{prefix}{t}" for t in texts], normalize_embeddings=True
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._encode(texts, "passage: ")]

    def embed_query(self, text: str):
        return self._encode([text], "query: ")[0]

    def embed_passages(self, texts: list[str]):
        return self._encode(texts, "passage: ")


class EmbeddingRetriever(BaseRetriever):
    """Dense cross-lingual retriever over a shared multilingual-e5 space."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        *,
        min_score: float | None = None,
    ) -> None:
        self.embedder = embedder or MultilingualE5Embedder()
        if min_score is None:
            min_score = float(os.getenv("E5_MIN_SCORE", "0.80"))
        self.min_score = min_score
        self._chunks: list[Chunk] = []
        self._matrix = None  # lazily-built (n, d) normalized embedding matrix

    def ensure_loaded(self) -> None:
        ensure = getattr(self.embedder, "ensure_loaded", None)
        if ensure is not None:
            ensure()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        import numpy as np

        vectors = self.embedder.embed_passages([c.text for c in chunks])
        vectors = np.asarray(vectors, dtype="float32")
        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._matrix is None or top_k <= 0:
            return []
        import numpy as np

        q = np.asarray(self.embedder.embed_query(query), dtype="float32")
        scores = self._matrix @ q  # cosine (both normalized)
        order = np.argsort(-scores)[:top_k]
        out: list[RetrievedChunk] = []
        for i in order:
            score = float(scores[i])
            if score < self.min_score:  # relevance gate -> out-of-corpus refuses
                continue
            out.append(RetrievedChunk(chunk=self._chunks[i], score=score))
        return out
