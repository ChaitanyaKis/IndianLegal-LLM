"""Citation plumbing: the system prompt, prompt builder, and citation extraction.

This module encodes the trust property (CLAUDE.md §4) at the prompt boundary:

* ``SYSTEM_PROMPT`` instructs the model to cite a retrieved ``chunk_id`` or refuse.
* ``build_user_prompt`` lays out the retrieved sources with their ids.
* ``extract_cited_ids`` parses ``[chunk_id]`` markers out of the model's reply.
* ``to_citation`` turns a retrieved chunk into a :class:`Citation`.

Enforcement (dropping citations to non-retrieved ids, refusing when none remain)
lives in the Answerer; this module only provides the parsing primitives.
"""

from __future__ import annotations

import re

from ..schemas import Chunk, Citation, RetrievedChunk

# Returned verbatim by the Answerer whenever no valid citation survives filtering.
REFUSAL_MESSAGE = (
    "I can't provide a grounded answer to this question from the retrieved Indian "
    "legal sources. To avoid an ungrounded legal claim, I'm declining to answer."
)

SYSTEM_PROMPT = (
    "You are IndianLegal-LLM, an assistant for Indian law only (Supreme Court of "
    "India, High Courts, and Indian statutes/bare acts).\n"
    "\n"
    "RULES (non-negotiable):\n"
    "1. Answer ONLY using the numbered sources provided in the user message.\n"
    "2. Every factual or legal statement you make MUST be supported by a citation "
    "to the source's identifier, written in square brackets, e.g. the bracketed "
    "id shown next to each source.\n"
    "3. Cite ONLY identifiers that appear in the provided sources. Never invent an "
    "identifier and never cite from memory.\n"
    "4. If the provided sources do not support an answer — including any non-Indian "
    "-law or out-of-domain question — you MUST refuse and cite nothing.\n"
    "5. No ungrounded legal claims, ever."
)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Render the question and retrieved sources into the user prompt.

    Each source is printed as a header line ``[<chunk_id>] <court> — <title>``
    followed by the chunk text on the next line. The id is deliberately the first
    bracketed token of each source block so a model (including the StubLLM) can
    align its citation to a real, retrieved id.
    """
    lines = [f"Question: {question}", "", "Sources:"]
    if not retrieved:
        lines.append("(none)")
    else:
        for rc in retrieved:
            c = rc.chunk
            lines.append(f"[{c.chunk_id}] {c.court} - {c.title}")
            lines.append(c.text)
            lines.append("")  # blank line separates sources
    lines.append(
        "Instructions: Answer using ONLY the sources above and cite the "
        "square-bracketed identifier of each source you rely on. If the sources "
        "do not contain the answer, refuse and cite nothing."
    )
    return "\n".join(lines)


# chunk_id charset: letters, digits, and the separators we use (._:-). No brackets.
_CITED_ID_RE = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")


def extract_cited_ids(text: str) -> list[str]:
    """Return the bracketed ids cited in ``text``, de-duplicated, in first-seen order."""
    seen: dict[str, None] = {}
    for match in _CITED_ID_RE.findall(text):
        seen.setdefault(match, None)
    return list(seen)


def to_citation(chunk: Chunk) -> Citation:
    """Build a :class:`Citation` from a chunk that was actually retrieved."""
    return Citation(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        court=chunk.court,
        url=chunk.url,
    )
