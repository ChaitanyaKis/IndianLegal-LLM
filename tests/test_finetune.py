"""Tests for the QLoRA instruction-dataset builder (offline, no GPU).

The key guarantee: every generated training answer PASSES the Hour-4 citation
guard, so a model fine-tuned on this data produces guard-passing answers.
"""

from __future__ import annotations

import json

from indianlegal_llm.finetune import (
    build_instruction_examples,
    examples_to_chat_records,
    write_jsonl,
)
from indianlegal_llm.ingestion.stub import StubIngestor
from indianlegal_llm.processing.stub import StubProcessor
from indianlegal_llm.rag.citation import assess_citations


def _corpus_chunks():
    processor = StubProcessor()
    return [c for doc in StubIngestor().fetch() for c in processor.process(doc)]


def test_build_instruction_examples_shape():
    chunks = _corpus_chunks()
    examples = build_instruction_examples(chunks)
    assert examples
    assert len(examples) <= len(chunks)
    for ex in examples:
        assert ex.messages[0]["role"] == "system"
        assert ex.messages[1]["role"] == "user"
        assert ex.messages[-1]["role"] == "assistant"
        assert f"[{ex.chunk_id}]" in ex.answer  # cites the source it was built from
        assert '"' in ex.answer  # carries a quoted proposition


def test_every_training_answer_passes_the_citation_guard():
    chunks = _corpus_chunks()
    by_id = {c.chunk_id: c for c in chunks}
    examples = build_instruction_examples(chunks)
    assert examples
    for ex in examples:
        valid, reason = assess_citations(ex.answer, {ex.chunk_id: by_id[ex.chunk_id]})
        assert reason is None, f"training answer would be refused ({reason}): {ex.answer}"
        assert ex.chunk_id in valid


def test_examples_to_chat_records():
    examples = build_instruction_examples(_corpus_chunks())
    records = examples_to_chat_records(examples)
    assert len(records) == len(examples)
    assert all(set(r) == {"messages"} for r in records)


def test_write_jsonl_under_data(tmp_path):
    examples = build_instruction_examples(_corpus_chunks())
    path = tmp_path / "data" / "train.jsonl"
    count = write_jsonl(examples, str(path))
    assert count == len(examples)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == count
    record = json.loads(lines[0])
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][-1]["role"] == "assistant"
