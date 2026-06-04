"""The single place where implementations are wired together.

`build_pipeline()` is the ONLY sanctioned wiring point (CLAUDE.md). The CLI, API,
demo, and evaluation harness all call it; none of them construct components
directly. To swap a stub for a real implementation, change it here — every caller
picks up the change for free, and the Base* interfaces guarantee they still fit.
"""

from __future__ import annotations

from .config import Settings
from .ingestion.base import BaseIngestor
from .ingestion.stub import StubIngestor
from .model.base import BaseLLM
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
    ) -> None:
        self.answerer = answerer
        self.settings = settings
        #: Ingestion manifest (url/court/date/license per doc) — CLAUDE.md §3.
        self.manifest = manifest
        self.num_chunks = num_chunks

    def answer(self, question: str) -> Answer:
        """Answer a question, citation-grounded (or refuse)."""
        return self.answerer.answer(question)


def build_pipeline(
    settings: Settings | None = None,
    *,
    ingestor: BaseIngestor | None = None,
    processor: BaseProcessor | None = None,
    retriever: BaseRetriever | None = None,
    llm: BaseLLM | None = None,
) -> Pipeline:
    """Construct the end-to-end pipeline.

    Defaults are the pure-stdlib stubs, so the skeleton runs with zero
    dependencies and zero configuration. Any component can be overridden with a
    real implementation that satisfies the same Base* interface — that override
    is the intended extension mechanism.
    """
    settings = settings or Settings.from_env()
    ingestor = ingestor or StubIngestor()
    processor = processor or StubProcessor()
    retriever = retriever or InMemoryRetriever()
    llm = llm or StubLLM()

    manifest: list[dict] = []
    chunks = []
    for doc in ingestor.fetch():
        manifest.append(doc.manifest_entry())  # log every source (CLAUDE.md §3)
        chunks.extend(processor.process(doc))
    retriever.add(chunks)

    answerer = Answerer(retriever=retriever, llm=llm, settings=settings)
    return Pipeline(
        answerer=answerer,
        settings=settings,
        manifest=manifest,
        num_chunks=len(chunks),
    )
