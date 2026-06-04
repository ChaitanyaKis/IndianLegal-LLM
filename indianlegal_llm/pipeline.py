"""The single place where implementations are wired together.

`build_pipeline()` is the ONLY sanctioned wiring point (CLAUDE.md). The CLI, API,
demo, and evaluation harness all call it; none of them construct components
directly. To swap a stub for a real implementation, change it here — every caller
picks up the change for free, and the Base* interfaces guarantee they still fit.

Ingestor selection: the configured ingestor (``Settings.ingestor``, default
``aws-sc``) is resolved via the ingestion registry. If an auto-resolved REAL
ingestor cannot run — its optional dependency (s3fs/pyarrow/httpx) is missing, or
there is no network — `build_pipeline()` falls back to the offline
:class:`StubIngestor` with a clear warning, so the zero-dependency skeleton always
runs and the green-build invariant (CLAUDE.md §6) holds. An explicitly passed
``ingestor`` is never overridden (the eval harness relies on this).

LLM selection mirrors this: ``Settings.llm`` (default ``transformers``, serving
``Settings.base_model`` 4-bit on a CUDA GPU) is resolved via the model registry.
If the real backend can't load — no GPU, or the `model` extra is missing — it
falls back to the offline :class:`StubLLM` with a clear warning, so local/offline
dev still runs. An explicitly passed ``llm`` is never overridden (the eval
harness pins :class:`StubLLM` for determinism).
"""

from __future__ import annotations

import sys

from .config import Settings
from .ingestion._errors import PROGRAMMER_ERRORS
from .ingestion.base import BaseIngestor
from .ingestion.registry import get_ingestor
from .ingestion.stub import StubIngestor
from .model.base import BaseLLM
from .model.registry import get_llm
from .model.stub import StubLLM
from .processing.base import BaseProcessor
from .processing.stub import StubProcessor
from .rag.answerer import Answerer
from .rag.base import BaseRetriever
from .rag.retriever import InMemoryRetriever
from .schemas import Answer


class Pipeline:
    """A built, ready-to-query pipeline plus the provenance of what it indexed."""

    def __init__(
        self,
        answerer: Answerer,
        settings: Settings,
        manifest: list[dict],
        num_chunks: int,
        source: str,
    ) -> None:
        self.answerer = answerer
        self.settings = settings
        #: Ingestion manifest (url/court/date/license per doc) — CLAUDE.md §3.
        self.manifest = manifest
        self.num_chunks = num_chunks
        #: ``source_name`` of the ingestor actually used (after any fallback).
        self.source = source

    @property
    def llm_backend(self) -> str:
        """Class name of the LLM actually serving (after any fallback)."""
        return type(self.answerer.llm).__name__

    def answer(self, question: str) -> Answer:
        """Answer a question, citation-grounded (or refuse)."""
        return self.answerer.answer(question)


def _resolve_retriever(settings: Settings) -> BaseRetriever:
    """Resolve the retriever from VECTOR_BACKEND, falling back to lexical.

    "memory" (default) is the lexical :class:`InMemoryRetriever`. "e5" is the
    cross-lingual :class:`EmbeddingRetriever` (multilingual-e5); if its deps are
    missing it degrades to the lexical retriever so the skeleton always runs.
    """
    backend = (settings.vector_backend or "memory").strip().lower()
    if backend in ("memory", "lexical", "stub"):
        return InMemoryRetriever()
    if backend in ("e5", "embedding", "multilingual-e5", "vector"):
        try:
            from .rag.embedding_retriever import EmbeddingRetriever

            retriever = EmbeddingRetriever()
            retriever.ensure_loaded()  # imports sentence-transformers + loads e5
            return retriever
        except Exception as exc:  # missing extra / model load failure
            print(
                f"[indianlegal_llm] WARNING: vector backend '{settings.vector_backend}' "
                f"unavailable ({type(exc).__name__}: {exc}); falling back to the "
                f"lexical InMemoryRetriever. Install the rag extra for cross-lingual "
                f"retrieval (pip install -e .[rag]).",
                file=sys.stderr,
            )
            return InMemoryRetriever()
    return InMemoryRetriever()


def _resolve_llm(settings: Settings) -> BaseLLM:
    """Resolve the configured LLM, falling back to the stub if it can't load.

    Loading is attempted eagerly here (via ``ensure_loaded``) so unavailability —
    missing `model` extra, or no CUDA GPU (in which case weights are NOT
    downloaded, CLAUDE.md §5) — degrades to the offline StubLLM before answering,
    keeping the CLI runnable (CLAUDE.md §6).
    """
    if (settings.llm or "stub").strip().lower() == "stub":
        return StubLLM()
    try:
        llm = get_llm(
            settings.llm, base_model=settings.base_model, adapter=settings.adapter
        )
        ensure_loaded = getattr(llm, "ensure_loaded", None)
        if ensure_loaded is not None:
            ensure_loaded()
        return llm
    except Exception as exc:  # missing extra / no GPU / load failure
        print(
            f"[indianlegal_llm] WARNING: LLM '{settings.llm}' unavailable "
            f"({type(exc).__name__}: {exc}); falling back to StubLLM. Use LLM=stub "
            f"for local dev, or a cloud CUDA GPU with the model extra "
            f"(pip install -e .[model]).",
            file=sys.stderr,
        )
        return StubLLM()


def _ingest(ingestor: BaseIngestor, processor: BaseProcessor) -> tuple[list[dict], list]:
    """Run ingestion -> processing, returning (manifest, chunks) as fresh lists.

    Documents are de-duplicated by ``doc_id`` (first occurrence wins): real
    corpora can repeat a case id, and a citation must map to exactly one source
    (CLAUDE.md §4). The retriever still enforces uniqueness as a backstop.
    """
    manifest: list[dict] = []
    chunks: list = []
    seen_doc_ids: set[str] = set()
    duplicates = 0
    for doc in ingestor.fetch():
        if doc.doc_id in seen_doc_ids:
            duplicates += 1
            continue
        seen_doc_ids.add(doc.doc_id)
        manifest.append(doc.manifest_entry())  # log every source (CLAUDE.md §3)
        chunks.extend(processor.process(doc))
    if duplicates:
        print(
            f"[indianlegal_llm] note: skipped {duplicates} duplicate doc_id(s) "
            "during ingestion (first occurrence kept).",
            file=sys.stderr,
        )
    return manifest, chunks


def build_pipeline(
    settings: Settings | None = None,
    *,
    ingestor: BaseIngestor | None = None,
    processor: BaseProcessor | None = None,
    retriever: BaseRetriever | None = None,
    llm: BaseLLM | None = None,
) -> Pipeline:
    """Construct the end-to-end pipeline.

    With no overrides, the ingestor comes from ``settings.ingestor`` and every
    other component is a pure-stdlib stub. A real ingestor that cannot run falls
    back to the stub (see module docstring); other components are overridden by
    passing a real implementation that satisfies the same Base* interface.
    """
    settings = settings or Settings.from_env()
    processor = processor or StubProcessor()
    # Resolve the retriever (lexical by default, e5 if configured) and the LLM
    # (real by default) only when not explicitly supplied; the eval harness passes
    # InMemoryRetriever() + StubLLM() so it is deterministic and offline.
    retriever = retriever if retriever is not None else _resolve_retriever(settings)
    llm = llm if llm is not None else _resolve_llm(settings)

    auto = ingestor is None
    if auto:
        ingestor = get_ingestor(settings.ingestor, limit=settings.ingest_limit)

    def _index(ing: BaseIngestor, ret: BaseRetriever) -> tuple[list[dict], list]:
        manifest, chunks = _ingest(ing, processor)
        ret.add(chunks)  # inside the fallback scope: index errors degrade too
        return manifest, chunks

    if auto and not isinstance(ingestor, StubIngestor):
        try:
            manifest, chunks = _index(ingestor, retriever)
        except Exception as exc:  # missing extra / network / data
            # Always surface real code defects; in strict mode surface everything.
            if settings.ingestor_strict or isinstance(exc, PROGRAMMER_ERRORS):
                raise
            print(
                f"[indianlegal_llm] WARNING: ingestor '{settings.ingestor}' "
                f"unavailable ({type(exc).__name__}: {exc}); falling back to "
                f"StubIngestor for the offline skeleton. Set INGESTOR_STRICT=1 to "
                f"make this a hard error, or run "
                f"`python -m indianlegal_llm.ingestion --source {settings.ingestor}` "
                f"for real ingestion.",
                file=sys.stderr,
            )
            ingestor = StubIngestor()
            retriever = InMemoryRetriever()  # fresh: never reuse a partially-filled index
            manifest, chunks = _index(ingestor, retriever)
    else:
        manifest, chunks = _index(ingestor, retriever)

    answerer = Answerer(retriever=retriever, llm=llm, settings=settings)
    return Pipeline(
        answerer=answerer,
        settings=settings,
        manifest=manifest,
        num_chunks=len(chunks),
        source=ingestor.source_name,
    )
