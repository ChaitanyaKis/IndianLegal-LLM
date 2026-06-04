"""Stub LLM: deterministic, dependency-free, and faithful to the trust property.

The stub never invents knowledge. It reads the sources laid out in the user
prompt and:

* if at least one source is present, it answers by grounding in — and citing —
  the FIRST (top-ranked) source;
* if no source is present, it produces a refusal and cites nothing.

Because it only ever cites the first source actually given to it, it can never
produce a citation that the Answerer would have to drop, and its prose is a
verbatim echo of that source (so it is faithful by construction). A real LLM
swapped in behind :class:`BaseLLM` still has its citations validated by the
Answerer (attribution), but its prose is not — content faithfulness is the real
model's responsibility (see the Answerer docstring and docs/ROADMAP.md).
"""

from __future__ import annotations

import re

from .base import BaseLLM

# Matches a source block header "[<id>] ..." followed by the chunk text line.
_SOURCE_RE = re.compile(r"\[([A-Za-z0-9_.:\-]+)\][^\n]*\n([^\n]+)")

_NO_SOURCE_REPLY = (
    "The provided sources do not support an answer to this question, so I will "
    "not answer. (No citation.)"
)


class StubLLM(BaseLLM):
    """Deterministic placeholder model. Cites the first source or refuses."""

    model_id = "stub-llm"

    #: Max characters of the source text to echo back into the answer.
    snippet_chars = 300

    def generate(self, system: str, user: str) -> str:
        match = _SOURCE_RE.search(user)
        if match is None:
            return _NO_SOURCE_REPLY

        chunk_id, source_text = match.group(1), match.group(2).strip()
        snippet = source_text[: self.snippet_chars].rstrip()
        return (
            f"Based on the retrieved Indian legal authority: {snippet} [{chunk_id}]"
        )
