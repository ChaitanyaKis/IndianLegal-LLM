"""Answerer: the component that enforces the trust property (CLAUDE.md §4).

Flow:  retrieve -> build prompt -> LLM -> keep only citations whose id was actually
retrieved -> refuse if none remain.

Guarantee: a non-refused :class:`Answer` always carries at least one citation to
a chunk that was actually retrieved for this query. The Answerer enforces
*attribution* — every returned claim is tied to a retrieved source, and citations
to non-retrieved ids are dropped (refusing if none survive).

Boundary (important): this is a citation-*existence* check, not a content-
*faithfulness* check. It does not compare the model's prose to the cited chunk's
text, so it cannot by itself stop a real LLM from writing a sentence the cited
source does not support. The shipped :class:`~indianlegal_llm.model.stub.StubLLM`
is faithful by construction; a real model wired in behind :class:`BaseLLM` should
add a faithfulness step (see docs/ROADMAP.md, Milestone 4) on top of this
attribution guarantee.
"""

from __future__ import annotations

from ..config import Settings
from ..model.base import BaseLLM
from ..schemas import Answer
from .base import BaseRetriever
from .citation import (
    SYSTEM_PROMPT,
    assess_citations,
    build_user_prompt,
    refusal_text,
    to_citation,
)


class Answerer:
    """Citation-grounded question answerer."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    def _refuse(self, question: str, reason: str) -> Answer:
        return Answer(
            question=question,
            text=refusal_text(reason),
            citations=[],
            refused=True,
        )

    def answer(self, question: str) -> Answer:
        retrieved = self.retriever.retrieve(question, self.settings.top_k)
        user_prompt = build_user_prompt(question, retrieved)
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        # Map retrieved ids -> their chunks; only these may be cited. The Citation
        # objects below are built purely from this trusted metadata, never from the
        # model's free text (a fabricated case/citation is structurally impossible).
        retrieved_by_id = {rc.chunk.chunk_id: rc.chunk for rc in retrieved}

        # Guard 1 (retrieved-set): only cited ids actually retrieved may stand.
        # Guard 2 (quote grounding, global): every quoted proposition must be
        # verbatim in some retrieved chunk. Refuse with a clear reason otherwise.
        valid_cited_ids, refusal_reason = assess_citations(raw, retrieved_by_id)
        if refusal_reason is not None:
            return self._refuse(question, refusal_reason)

        citations = [to_citation(retrieved_by_id[cid]) for cid in valid_cited_ids]
        return Answer(
            question=question,
            text=raw,
            citations=citations,
            refused=False,
        )
