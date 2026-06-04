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
"""

from __future__ import annotations

import sys

from .config import Settings
from .ingestion.base import BaseIngestor
from .ingestion.registry import get_ingestor
from .ingestion.stub import StubIngestor
from .model.base import BaseLLM
from .model.stub import StubLLM
from .processing.base import BaseProcessor
from .processing.stub import StubProcessor
from .rag.answerer import Answerer
from .rag.base import BaseRetriever
from .rag.retriever import InMemoryRetriever
from .schemas import Answer

# Exception types that signal a programmer error rather than a degradable runtime
# condition (missing extra / network / data). These propagate instead of silently
# falling back to the stub, so a real code defect can't masquerade as "ingestor
# unavailable" and pass the green-build gate.
_PROGRAMMER_ERRORS = (AttributeError, TypeError, NameError, KeyError, IndexError)


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

    def answer(self, question: str) -> Answer:
        """Answer a question, citation-grounded (or refuse)."""
        return self.answerer.answer(question)


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
    retriever = retriever or InMemoryRetriever()
    llm = llm or StubLLM()

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
        except Exception as exc:  # missing extra / network / data — keep skeleton alive
            if isinstance(exc, _PROGRAMMER_ERRORS):
                raise  # a real code defect must surface, not masquerade as "unavailable"
            print(
                f"[indianlegal_llm] WARNING: ingestor '{settings.ingestor}' "
                f"unavailable ({type(exc).__name__}: {exc}); falling back to "
                f"StubIngestor for the offline skeleton. Run "
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
