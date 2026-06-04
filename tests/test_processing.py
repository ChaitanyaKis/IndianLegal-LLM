"""Tests for paragraph-aware chunking (pinpoint-citation support).

Indian judgments are paragraph-numbered; chunks must keep each chunk's paragraph
span (para_start, para_end) and prefer paragraph boundaries.
"""

from __future__ import annotations

from pathlib import Path

from indianlegal_llm.ingestion.stub import StubIngestor
from indianlegal_llm.processing._paragraphs import segment_paragraphs
from indianlegal_llm.processing.stub import StubProcessor
from indianlegal_llm.schemas import RawDoc

FIXTURES = Path(__file__).parent / "fixtures"


def _doc(text: str) -> RawDoc:
    return RawDoc(
        doc_id="test-judgment",
        title="Test v. State",
        court="Supreme Court of India",
        date="2024-01-01",
        url="https://example.org/doc/1",
        license="government-work",
        text=text,
        language="en",
    )


def test_segment_paragraphs_detects_numbers_and_pilcrow():
    text = (FIXTURES / "numbered_judgment.txt").read_text(encoding="utf-8")
    paras = segment_paragraphs(text)
    numbers = [p.number for p in paras]
    assert None in numbers  # the case-title preamble is unnumbered
    assert [n for n in numbers if n is not None] == [1, 2, 3, 4, 5, 6]  # incl. ¶ 5


def test_paragraph_spans_are_captured_on_fixture():
    text = (FIXTURES / "numbered_judgment.txt").read_text(encoding="utf-8")
    chunks = StubProcessor(chunk_words=30).process(_doc(text))
    assert chunks

    # Every chunk records a paragraph span in metadata.
    for chunk in chunks:
        assert "para_start" in chunk.metadata
        assert "para_end" in chunk.metadata

    spans = [
        (c.metadata["para_start"], c.metadata["para_end"])
        for c in chunks
        if c.metadata["para_start"] is not None
    ]
    assert spans, "no paragraph spans captured"
    for start, end in spans:
        assert start <= end

    # Paragraphs 1..6 are covered (the pilcrow paragraph 5 included).
    covered: set[int] = set()
    for start, end in spans:
        covered.update(range(start, end + 1))
    assert covered == {1, 2, 3, 4, 5, 6}
    assert any(start <= 5 <= end for start, end in spans)


def test_chunks_do_not_split_a_paragraph_midway():
    text = (FIXTURES / "numbered_judgment.txt").read_text(encoding="utf-8")
    chunks = StubProcessor(chunk_words=30).process(_doc(text))
    # A distinctive phrase from paragraph 3 must live wholly inside one chunk.
    phrase = "audi alteram partem is a facet of Article 14"
    holders = [c for c in chunks if phrase in c.text]
    assert len(holders) == 1


def test_oversized_paragraph_is_windowed_but_keeps_its_number():
    big = "1. " + " ".join(f"word{i}" for i in range(200))
    chunks = StubProcessor(chunk_words=50).process(_doc(big))
    assert len(chunks) > 1  # windowed
    for chunk in chunks:
        assert chunk.metadata["para_start"] == 1
        assert chunk.metadata["para_end"] == 1


def test_stub_judgments_carry_paragraph_spans():
    """The real stub corpus is paragraph-numbered, so spans flow end-to-end."""
    processor = StubProcessor()
    for doc in StubIngestor().fetch():
        chunks = processor.process(doc)
        numbered = [c for c in chunks if c.metadata.get("para_start") is not None]
        assert numbered, f"{doc.doc_id} produced no numbered paragraph spans"


def test_unnumbered_text_yields_none_spans():
    chunks = StubProcessor().process(_doc("Just some prose with no paragraph numbers."))
    assert chunks
    assert all(c.metadata["para_start"] is None for c in chunks)
