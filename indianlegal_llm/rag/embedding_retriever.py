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
import sys

from ..schemas import Chunk, RetrievedChunk
from .base import BaseEmbedder, BaseRetriever

# intfloat/multilingual-e5-* are MIT-licensed and cover ~100 languages incl. the
# major Indian scripts. -small is CPU-friendly; override the HF id via the
# EMBEDDING_MODEL_NAME env var (the variable the code below actually reads).
_DEFAULT_E5_MODEL = "intfloat/multilingual-e5-small"

# Data-parallel embedding (encode_multi_process across all CUDA devices) only pays
# off for large corpora; below this many passages a single device is faster after
# pool-startup overhead. e5 fits on one 16 GB GPU, so this is THROUGHPUT sharding,
# not model sharding. Override via the E5_MULTI_GPU_MIN_CHUNKS env var.
_DEFAULT_MULTI_GPU_MIN_TEXTS = 50_000


def cuda_device_count() -> int:
    """Number of visible CUDA devices (0 if torch is absent or no GPU is present)."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch absent in offline/CI
        return 0
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # pragma: no cover - driver/runtime hiccup
        return 0


def _should_shard(n_texts: int, device_count: int, threshold: int) -> bool:
    """True iff multi-GPU data-parallel encoding is worth it: more than one device,
    a positive threshold, and enough passages to amortize the pool startup cost."""
    return device_count > 1 and threshold > 0 and n_texts >= threshold


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:  # garbage env value -> safe default
        return default


class MultilingualE5Embedder(BaseEmbedder):
    """Sentence-Transformers wrapper for multilingual-e5 (e5 needs role prefixes).

    Bulk passage encoding (:meth:`embed_passages`) transparently shards across all
    CUDA devices via a multi-process pool when the corpus is large enough and more
    than one GPU is visible; otherwise it encodes on a single device. The output is
    identical either way — a normalized ``(n, d)`` matrix in input order — so this
    is a throughput optimization behind the unchanged :class:`BaseEmbedder` contract.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        multi_gpu_min_texts: int | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", _DEFAULT_E5_MODEL
        )
        self.multi_gpu_min_texts = (
            multi_gpu_min_texts
            if multi_gpu_min_texts is not None
            else _env_int("E5_MULTI_GPU_MIN_CHUNKS", _DEFAULT_MULTI_GPU_MIN_TEXTS)
        )
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
        """Encode passages, data-parallel across GPUs for large corpora.

        Returns a normalized ``(n, d)`` matrix in input order. Shards across all
        CUDA devices only when ``cuda_device_count() > 1`` and ``len(texts)`` clears
        ``multi_gpu_min_texts``; otherwise (and on any pool failure) it encodes on a
        single device. Logs which path ran and the device count actually used.
        """
        device_count = cuda_device_count()
        if _should_shard(len(texts), device_count, self.multi_gpu_min_texts):
            sharded = self._encode_passages_multi_gpu(texts, device_count)
            if sharded is not None:
                return sharded
        print(
            f"[e5] single-device encode: {len(texts)} passages "
            f"(visible CUDA devices={device_count})",
            file=sys.stderr,
        )
        return self._encode(texts, "passage: ")

    def _encode_passages_multi_gpu(self, texts: list[str], device_count: int):
        """Data-parallel passage encoding across all CUDA devices.

        Returns a normalized ``(n, d)`` float32 matrix in input order, or ``None``
        to signal a clean fallback to single-device encoding (e.g. the
        multi-process pool could not start). The pool is always torn down.
        """
        self.ensure_loaded()
        target_devices = [f"cuda:{i}" for i in range(device_count)]
        pool = None
        try:
            import numpy as np

            # NOTE: start_multi_process_pool uses the "spawn" start method on ALL
            # platforms (incl. Linux), so its workers re-import the entry module.
            # This must therefore be driven from a script under an
            # `if __name__ == "__main__"` guard (scripts/kaggle_run.py is) — never
            # from a guard-less notebook cell, which would deadlock the spawn.
            pool = self._model.start_multi_process_pool(target_devices=target_devices)
            print(
                f"[e5] multi-GPU data-parallel encode: {len(texts)} passages across "
                f"{len(target_devices)} devices {target_devices}",
                file=sys.stderr,
            )
            prefixed = [f"passage: {t}" for t in texts]
            # encode_multi_process is the stable multi-GPU call in ST 2.x-4.x and
            # reassembles results IN INPUT ORDER; newer releases deprecate it for
            # encode(..., pool=pool). Prefer whichever the installed version exposes.
            encode_mp = getattr(self._model, "encode_multi_process", None)
            if encode_mp is not None:
                emb = encode_mp(prefixed, pool)
            else:  # pragma: no cover - only on ST versions without the old method
                emb = self._model.encode(prefixed, pool=pool)
            emb = np.asarray(emb, dtype="float32")
            # Neither call normalizes by default, so normalize here to match the
            # single-device path exactly (cosine == dot, identical min_score gate).
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return emb / norms
        except Exception as exc:  # noqa: BLE001 - any failure -> single-device path
            print(
                f"[e5] WARNING: multi-GPU pool unavailable "
                f"({type(exc).__name__}: {exc}); using single-device encode.",
                file=sys.stderr,
            )
            return None
        finally:
            if pool is not None:
                try:
                    self._model.stop_multi_process_pool(pool)
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass


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
