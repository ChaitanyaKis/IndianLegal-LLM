"""Indic-language path tests (offline, deterministic).

The differentiation: an Indian-language question is EXPLAINED in that language, but
the verbatim QUOTE stays in the source language (English) so the citation guard
still verifies it. Cross-lingual *retrieval* is the e5/quality tier; here we prove
the language detection, the prompt directive, and — critically — that the guard
stays intact cross-lingually (English quote grounds; a translated quote is refused).
"""

from __future__ import annotations

import pytest

from indianlegal_llm.rag.citation import (
    REASON_UNGROUNDED,
    assess_citations,
    build_user_prompt,
)
from indianlegal_llm.rag.lang import detect_language, language_name
from indianlegal_llm.schemas import Chunk, RetrievedChunk

# A Devanagari (Hindi) question and a Telugu question.
HINDI_Q = "क्या भारत में निजता एक मौलिक अधिकार है?"
TELUGU_Q = "ఫ్రాన్స్ రాజధాని ఏది?"

_CHUNK = Chunk(
    chunk_id="puttaswamy-2017::0",
    doc_id="puttaswamy-2017",
    text="The right to privacy is a fundamental right protected under Article 21 of the Constitution.",
    title="K. S. Puttaswamy v. Union of India",
    court="Supreme Court of India",
    url="https://indiankanoon.org/doc/91938676/",
    license="government-work",
    metadata={"nc_display": "2017 INSC 1", "para_start": 297, "para_end": 297},
)
_RETRIEVED = {_CHUNK.chunk_id: _CHUNK}


def test_detect_language():
    assert detect_language(HINDI_Q) == "hi"
    assert detect_language(TELUGU_Q) == "te"
    assert detect_language("Is privacy a fundamental right in India?") == "en"
    assert detect_language("") == "en"
    assert language_name("hi") == "Hindi"
    assert language_name("te") == "Telugu"


def test_user_prompt_adds_language_directive_only_for_indic():
    en = build_user_prompt("Is privacy a fundamental right?", [RetrievedChunk(_CHUNK, 1.0)])
    assert "Respond in" not in en
    hi = build_user_prompt(HINDI_Q, [RetrievedChunk(_CHUNK, 1.0)], answer_language="hi")
    assert "Respond in Hindi" in hi
    assert "VERBATIM in English" in hi


def test_cross_lingual_answer_with_english_quote_passes_guard():
    """A Hindi explanation + an English verbatim quote + [chunk_id] is grounded."""
    answer = (
        "निजता एक मौलिक अधिकार है। न्यायालय ने कहा: "
        '"The right to privacy is a fundamental right" [puttaswamy-2017::0]'
    )
    valid, reason = assess_citations(answer, _RETRIEVED)
    assert reason is None
    assert "puttaswamy-2017::0" in valid


def test_translated_quote_is_refused_cross_lingually():
    """A quote TRANSLATED into the user's language is not verbatim -> refused."""
    answer = (
        'न्यायालय ने कहा: "निजता एक मौलिक अधिकार है" [puttaswamy-2017::0]'
    )  # the "quote" is Hindi, not the English source text
    valid, reason = assess_citations(answer, _RETRIEVED)
    assert valid == []
    assert reason == REASON_UNGROUNDED


def test_indic_question_refuses_on_lexical_retriever():
    """On the stub/lexical retriever an Indic question retrieves nothing -> refuse
    (the deterministic gate's out-of-corpus refusal, no cross-lingual embedder)."""
    from indianlegal_llm.config import Settings
    from indianlegal_llm.ingestion.stub import StubIngestor
    from indianlegal_llm.model.stub import StubLLM
    from indianlegal_llm.pipeline import build_pipeline
    from indianlegal_llm.rag.retriever import InMemoryRetriever

    pipe = build_pipeline(
        Settings(), ingestor=StubIngestor(), llm=StubLLM(), retriever=InMemoryRetriever()
    )
    answer = pipe.answer(TELUGU_Q)
    assert answer.refused
    assert answer.citations == []
