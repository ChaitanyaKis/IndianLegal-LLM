"""Adversarial tests for the citation guard (CLAUDE.md §4).

Every case must end in a refusal or have the bad citation stripped; a citation is
ALWAYS built from the retrieved chunk's trusted metadata, never from model text.
No ungrounded legal claim is ever returned (hallucinations stays 0).
"""

from __future__ import annotations

import pytest

from indianlegal_llm.app.cli import format_answer
from indianlegal_llm.config import Settings
from indianlegal_llm.model.base import BaseLLM
from indianlegal_llm.rag.answerer import Answerer
from indianlegal_llm.rag.citation import to_citation
from indianlegal_llm.rag.retriever import InMemoryRetriever
from indianlegal_llm.schemas import Chunk, Citation

REAL_TEXT = (
    "The right to privacy is a fundamental right protected under Article 21 of the "
    "Constitution, as part of the right to life and personal liberty."
)
QUESTION = "privacy fundamental right Article 21"


def _retriever() -> InMemoryRetriever:
    retriever = InMemoryRetriever()
    retriever.add(
        [
            Chunk(
                chunk_id="real::0",
                doc_id="real-doc",
                text=REAL_TEXT,
                title="Real Case v. Union of India",
                court="Supreme Court of India",
                url="https://indiankanoon.org/doc/1/",
                license="government-work",
                metadata={"nc_display": "2017 INSC 1", "para_start": 297, "para_end": 297},
            )
        ]
    )
    return retriever


class _ScriptedLLM(BaseLLM):
    """An LLM that returns a fixed (possibly adversarial) string."""

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, system: str, user: str) -> str:
        return self._output


def _answer(output: str):
    answerer = Answerer(retriever=_retriever(), llm=_ScriptedLLM(output), settings=Settings())
    return answerer.answer(QUESTION)


# --------------------------------------------------------------------------- #
# Citation completeness (neutral citation threaded from metadata)
# --------------------------------------------------------------------------- #
def test_reference_renders_title_neutral_citation_and_pinpoint():
    cite = Citation(
        chunk_id="sc-2017-INSC-1::0",
        doc_id="sc-2017-INSC-1",
        title="K. S. Puttaswamy v. Union of India",
        court="Supreme Court of India",
        url="s3://bucket/doc",
        neutral_citation="2017 INSC 1",
        para_start=297,
        para_end=297,
    )
    assert cite.reference == "K. S. Puttaswamy v. Union of India, 2017 INSC 1 ¶ 297"


def test_to_citation_threads_neutral_citation_from_metadata():
    chunk = Chunk(
        chunk_id="sc-x::0",
        doc_id="sc-x",
        text="...",
        title="A v. B",
        court="Supreme Court of India",
        url="u",
        license="government-work",
        metadata={"nc_display": "2020 INSC 5", "para_start": 12, "para_end": 14},
    )
    cite = to_citation(chunk)
    assert cite.neutral_citation == "2020 INSC 5"
    assert cite.reference == "A v. B, 2020 INSC 5 ¶ 12-14"


# --------------------------------------------------------------------------- #
# Adversarial: the four required attacks
# --------------------------------------------------------------------------- #
def test_fabricated_case_and_citation_in_prose_never_enter_the_citation():
    out = (
        "In Fabricated v. Imaginary, 9999 INSC 9999, the Court held that privacy "
        "is protected. [real::0]"
    )
    answer = _answer(out)
    assert not answer.refused  # cites a real retrieved source
    assert len(answer.citations) == 1
    cite = answer.citations[0]
    # Citation comes from trusted metadata, NOT the model's fabricated prose.
    assert cite.title == "Real Case v. Union of India"
    assert cite.neutral_citation == "2017 INSC 1"
    rendered = format_answer(answer)
    # The fabricated case name / neutral citation never appear in the citation line.
    citation_line = [ln for ln in rendered.splitlines() if ln.strip().startswith("- [real::0]")][0]
    assert "Fabricated" not in citation_line
    assert "9999 INSC 9999" not in citation_line


def test_citing_a_non_retrieved_chunk_id_is_refused():
    answer = _answer("Privacy is protected. [ghost::99]")
    assert answer.refused
    assert answer.citations == []


def test_quoting_a_proposition_absent_from_the_chunk_is_refused():
    out = 'The Court held that "the moon is made of green cheese and pigs fly". [real::0]'
    answer = _answer(out)
    assert answer.refused  # ungrounded quote -> citation dropped -> none survive
    assert answer.citations == []


def test_answer_with_no_citation_is_refused():
    answer = _answer("Privacy is certainly a fundamental right, take my word for it.")
    assert answer.refused
    assert answer.citations == []


# --------------------------------------------------------------------------- #
# A genuinely grounded quote survives (guard is not over-eager)
# --------------------------------------------------------------------------- #
def test_grounded_quote_survives():
    out = 'The Court held that "a fundamental right protected under Article 21". [real::0]'
    answer = _answer(out)
    assert not answer.refused
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "real::0"


def _two_source_answerer(output: str) -> Answerer:
    retriever = InMemoryRetriever()
    retriever.add(
        [
            Chunk(
                chunk_id="a::0",
                doc_id="doc-a",
                text="Article 14 guarantees equality before the law to every person.",
                title="A v. State",
                court="Supreme Court of India",
                url="u-a",
                license="government-work",
                metadata={"nc_display": "2018 INSC 2", "para_start": 1, "para_end": 1},
            ),
            Chunk(
                chunk_id="b::0",
                doc_id="doc-b",
                text="The basic structure doctrine limits the amending power of Parliament.",
                title="B v. State",
                court="Supreme Court of India",
                url="u-b",
                license="government-work",
                metadata={"nc_display": "1973 INSC 3", "para_start": 2, "para_end": 2},
            ),
        ]
    )
    return Answerer(retriever=retriever, llm=_ScriptedLLM(output), settings=Settings())


def test_ungrounded_quote_anywhere_refuses_even_with_a_valid_citation():
    """A fabricated quote riding alongside a valid citation refuses the whole answer."""
    out = (
        'Equality before the law is guaranteed. [a::0] '
        'It also held "unicorns are tax-exempt under Article 99". [b::0]'
    )
    answer = _two_source_answerer(out).answer(
        "equality before the law basic structure Parliament"
    )
    assert answer.refused
    assert answer.citations == []


def test_genuine_multi_source_quote_is_not_falsely_refused():
    """A verbatim quote from one retrieved chunk, citing both, must NOT refuse."""
    out = (
        'The Court noted "the basic structure doctrine limits the amending power '
        'of Parliament". [a::0] [b::0]'
    )
    answer = _two_source_answerer(out).answer(
        "equality before the law basic structure Parliament"
    )
    assert not answer.refused
    assert {c.chunk_id for c in answer.citations} == {"a::0", "b::0"}


# --------------------------------------------------------------------------- #
# Red-team evasions (from the adversarial workflow) — all must refuse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,output",
    [
        # Negation inserted into an otherwise-verbatim quote (inverts the holding).
        ("negation", 'The Court held "the right to privacy is not a fundamental right under Article 21 of the Constitution". [real::0]'),
        # Word-order inversion ("NOT protected").
        ("reorder", 'The Court held "the right to privacy is a fundamental right NOT protected under Article 21". [real::0]'),
        # Token-stuffed fabrication padded with chunk words.
        ("stuffed", 'It held "the right to privacy fundamental Article 21 Constitution is hereby abolished forever". [real::0]'),
        # Dangling opening quote (no closing) to evade extraction.
        ("dangling", 'The Court held: "Parliament repealed Article 21 and privacy is abolished [real::0]'),
        # Citation marker placed inside the quotation to split it.
        ("marker-inside", '"Article 21 was struck down [real::0] and privacy is no longer a fundamental right" [real::0]'),
        # Fabricated quote in trailing text after the last marker.
        ("trailing", 'Privacy is protected [real::0]. "Article 21 was repealed and privacy abolished in 2040."'),
        # Mismatched quote glyphs (curly-open + straight-close).
        ("mismatched-glyph", 'The Court held “privacy may be suspended during any emergency without review". [real::0]'),
        # Guillemets around a fabricated proposition.
        ("guillemets", 'The Court held «privacy is abolished and Article 21 does not apply». [real::0]'),
        # Combining diacritic hiding the decisive negation token.
        ("combining-mark", 'The Court held "the right to privacy is nót a fundamental right under Article 21 of the Constitution". [real::0]'),
    ],
)
def test_adversarial_quote_evasions_are_refused(name, output):
    answer = _answer(output)
    assert answer.refused, f"{name} attack was NOT refused"
    assert answer.citations == []
