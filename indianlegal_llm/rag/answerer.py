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
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_cited_ids,
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

    def answer(self, question: str) -> Answer:
        retrieved = self.retriever.retrieve(question, self.settings.top_k)
        user_prompt = build_user_prompt(question, retrieved)
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        # Map retrieved ids -> their chunks; only these may be cited.
        retrieved_by_id = {rc.chunk.chunk_id: rc.chunk for rc in retrieved}
        valid_cited_ids = [
            cid for cid in extract_cited_ids(raw) if cid in retrieved_by_id
        ]

        # Trust property: no valid citation -> refuse. No ungrounded claims, ever.
        if not valid_cited_ids:
            return Answer(
                question=question,
                text=REFUSAL_MESSAGE,
                citations=[],
                refused=True,
            )

        citations = [to_citation(retrieved_by_id[cid]) for cid in valid_cited_ids]
        return Answer(
            question=question,
            text=raw,
            citations=citations,
            refused=False,
        )
