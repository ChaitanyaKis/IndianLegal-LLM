"""Build a citation-grounded instruction dataset from the processed corpus.

Each example trains the model to answer a question by grounding in a SHORT
VERBATIM quote from the retrieved chunk, followed by the ``[chunk_id]`` marker and
the paragraph pinpoint — exactly the contract the Hour-4 citation guard enforces
(``rag.citation.assess_citations``). So a model trained on this data produces
answers that PASS the guard rather than getting refused.

Generation method (documented, reproducible, license-clean):
- Examples are built TEMPLATE-based directly from the processed corpus chunks
  (themselves derived from license-clean sources, CLAUDE.md §3). No external
  teacher model is used, so there is no extra licensing/cost surface.
- The quoted proposition is a verbatim span of the cited chunk, so every example
  is grounded BY CONSTRUCTION (the test asserts each passes ``assess_citations``).
- Questions are templated from the chunk's court/title metadata.

The dataset is built and consumed IN THE CLOUD and written under the gitignored
``data/`` — never committed (CLAUDE.md §5). A richer LLM-distilled question set is
a documented future enhancement; the template method is the clean baseline.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..rag.citation import SYSTEM_PROMPT, build_user_prompt
from ..schemas import Chunk, RetrievedChunk

_PILCROW = chr(0x00B6)  # ¶
_LEADING_PARA_NUM = re.compile(r"^\s*\d+[.)]?\s+")

# Question templates derived from corpus metadata (license-clean, reproducible).
_QUESTION_TEMPLATES = (
    "What did the {court} hold in {title}?",
    "Summarise the relevant holding in {title}.",
    "What does {title} say on this point?",
    "According to {title}, what is the legal position here?",
)


@dataclass
class InstructionExample:
    """One supervised fine-tuning example (chat-formatted)."""

    question: str
    chunk_id: str
    doc_id: str
    answer: str
    messages: list[dict]


def _verbatim_quote(text: str, max_words: int = 28) -> str:
    """Return a clean, VERBATIM span of ``text`` for use as a quoted proposition.

    A leading paragraph number (e.g. "12.") is dropped; the remainder is a true
    substring of the chunk, so it grounds against that chunk by construction.
    """
    body = _LEADING_PARA_NUM.sub("", text.strip())
    return " ".join(body.split()[:max_words])


def _pinpoint(chunk: Chunk) -> str:
    start = chunk.metadata.get("para_start")
    end = chunk.metadata.get("para_end")
    if start is None:
        return ""
    if end is None or end == start:
        return f" {_PILCROW} {start}"
    return f" {_PILCROW} {start}-{end}"


def _answer(chunk: Chunk, quote: str) -> str:
    return (
        f'In {chunk.title}, the {chunk.court} observed: "{quote}" '
        f"[{chunk.chunk_id}]{_pinpoint(chunk)}."
    )


def build_instruction_examples(
    chunks: Iterable[Chunk], *, min_quote_words: int = 6
) -> list[InstructionExample]:
    """Build guard-passing (question -> grounded answer) examples from chunks.

    Chunks too short to yield a meaningful quote are skipped.
    """
    examples: list[InstructionExample] = []
    for index, chunk in enumerate(chunks):
        quote = _verbatim_quote(chunk.text)
        if len(quote.split()) < min_quote_words:
            continue
        template = _QUESTION_TEMPLATES[index % len(_QUESTION_TEMPLATES)]
        question = template.format(court=chunk.court, title=chunk.title)
        answer = _answer(chunk, quote)
        user_prompt = build_user_prompt(question, [RetrievedChunk(chunk=chunk, score=1.0)])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ]
        examples.append(
            InstructionExample(
                question=question,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                answer=answer,
                messages=messages,
            )
        )
    return examples


def examples_to_chat_records(examples: Iterable[InstructionExample]) -> list[dict]:
    """Return [{"messages": [...]}] records ready for an SFT chat template."""
    return [{"messages": ex.messages} for ex in examples]


def write_jsonl(examples: Iterable[InstructionExample], path: str) -> int:
    """Write chat records as JSONL (under the gitignored data/); return the count."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in examples_to_chat_records(examples):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
