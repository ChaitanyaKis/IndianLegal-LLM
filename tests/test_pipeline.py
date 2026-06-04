"""End-to-end tests for the IndianLegal-LLM walking skeleton.

These assert the acceptance criteria and, crucially, the trust property
(CLAUDE.md §4): no ungrounded legal claim is ever returned.
"""

from __future__ import annotations

import pytest

from indianlegal_llm.app.cli import format_answer
from indianlegal_llm.evaluation import harness
from indianlegal_llm.evaluation.cases import CASES
from indianlegal_llm.pipeline import build_pipeline
from indianlegal_llm.rag.citation import extract_cited_ids
from indianlegal_llm.schemas import Answer, Citation


@pytest.fixture(scope="module")
def pipeline():
    # Pin the offline stub corpus so these skeleton tests are deterministic and
    # network-independent regardless of the configured INGESTOR default.
    from indianlegal_llm.ingestion.stub import StubIngestor

    return build_pipeline(ingestor=StubIngestor())


def test_privacy_question_is_cited(pipeline):
    answer = pipeline.answer("Is privacy a fundamental right in India?")
    assert answer.is_grounded
    assert not answer.refused
    assert answer.citations
    assert any(c.doc_id == "puttaswamy-2017" for c in answer.citations)


def test_basic_structure_question_is_cited(pipeline):
    answer = pipeline.answer(
        "What is the basic structure doctrine of the Indian Constitution?"
    )
    assert answer.is_grounded
    assert any(c.doc_id == "kesavananda-1973" for c in answer.citations)


def test_out_of_domain_question_is_refused(pipeline):
    answer = pipeline.answer("What is the capital of France?")
    assert answer.refused
    assert answer.citations == []
    assert not answer.is_grounded


@pytest.mark.parametrize(
    "question",
    [
        "What is the structure of the European Union?",
        "What rights does the United States Bill of Rights guarantee?",
    ],
)
def test_incidental_overlap_out_of_domain_is_refused(pipeline, question):
    """Out-of-domain questions that share only an incidental token must refuse.

    This is the relevance-gate boundary: a single coincidental shared word must
    not be enough to retrieve (and thus answer from) the Indian-law corpus.
    NOTE: lexical overlap cannot catch questions that share *several* tokens with
    the corpus (e.g. foreign-constitution privacy questions); that is a known
    stub limitation closed by the semantic retriever (docs/ROADMAP.md M3).
    """
    answer = pipeline.answer(question)
    assert answer.refused, f"expected refusal for out-of-domain question: {question}"
    assert answer.citations == []


def test_eval_harness_is_green():
    metrics = harness.run()
    assert metrics["citation_accuracy"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["hallucinations"] == 0
    assert metrics["green"] is True


def test_harness_main_returns_zero():
    assert harness.main() == 0


def test_every_citation_is_a_retrieved_chunk(pipeline):
    """The trust property: a returned citation must be a chunk we retrieved."""
    for case in CASES:
        answer = pipeline.answer(case.question)
        retrieved = pipeline.answerer.retriever.retrieve(
            case.question, pipeline.settings.top_k
        )
        retrieved_ids = {rc.chunk.chunk_id for rc in retrieved}
        for citation in answer.citations:
            assert citation.chunk_id in retrieved_ids


def test_non_refused_answer_always_has_a_citation(pipeline):
    """No ungrounded legal claim, ever."""
    for case in CASES:
        answer = pipeline.answer(case.question)
        if not answer.refused:
            assert answer.citations, f"{case.case_id} answered without a citation"


def test_answerer_drops_fabricated_citations():
    """If the LLM cites an id that was never retrieved, the answer is refused."""
    from indianlegal_llm.config import Settings
    from indianlegal_llm.model.base import BaseLLM
    from indianlegal_llm.rag.answerer import Answerer
    from indianlegal_llm.rag.retriever import InMemoryRetriever

    class FabricatingLLM(BaseLLM):
        model_id = "fabricator"

        def generate(self, system: str, user: str) -> str:
            return "Privacy is protected. [totally-made-up-id]"

    # Retriever has no chunks -> nothing can match the fabricated id.
    answerer = Answerer(
        retriever=InMemoryRetriever(),
        llm=FabricatingLLM(),
        settings=Settings(),
    )
    answer = answerer.answer("anything")
    assert answer.refused
    assert answer.citations == []


def test_extract_cited_ids_dedupes_in_order():
    text = "see [a::1] and [b::2] and again [a::1]"
    assert extract_cited_ids(text) == ["a::1", "b::2"]


def test_citation_pinpoint_rendering():
    base = dict(chunk_id="d::0", doc_id="d", title="T", court="C", url="u")
    assert Citation(**base, para_start=297, para_end=297).pinpoint == "¶ 297"
    assert Citation(**base, para_start=12, para_end=14).pinpoint == "¶ 12-14"
    assert Citation(**base).pinpoint == ""  # unnumbered / preamble -> omitted


def test_cited_answer_carries_and_renders_paragraph_pinpoint(pipeline):
    answer = pipeline.answer("Is privacy a fundamental right in India?")
    assert answer.citations
    cite = answer.citations[0]
    assert cite.para_start is not None  # carried through to_citation from metadata
    assert cite.pinpoint.startswith("¶")
    rendered = format_answer(answer)
    assert cite.pinpoint in rendered  # pinpoint shown in CLI output


def test_unnumbered_citation_renders_without_pinpoint():
    answer = Answer(
        question="q",
        text="x [d::0]",
        citations=[Citation(chunk_id="d::0", doc_id="d", title="Some Case", court="SC", url="u")],
        refused=False,
    )
    rendered = format_answer(answer)
    assert "¶" not in rendered
    assert "Some Case" in rendered


def _chunk(chunk_id: str, doc_id: str = "d"):
    from indianlegal_llm.schemas import Chunk

    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="some text",
        title="t",
        court="Supreme Court of India",
        url="https://example.org",
        license="public record",
    )


def test_retriever_rejects_duplicate_chunk_id():
    """Duplicate ids would let a citation map to the wrong source (CLAUDE.md §4)."""
    from indianlegal_llm.rag.retriever import InMemoryRetriever

    retriever = InMemoryRetriever()
    retriever.add([_chunk("d::0", doc_id="d")])
    with pytest.raises(ValueError):
        retriever.add([_chunk("d::0", doc_id="other-doc")])


def test_retriever_rejects_invalid_chunk_id():
    from indianlegal_llm.rag.retriever import InMemoryRetriever

    retriever = InMemoryRetriever()
    for bad in ("", "  ", "has space", "weird]bracket["):
        with pytest.raises(ValueError):
            retriever.add([_chunk(bad)])


def test_parse_top_k_is_defensive():
    from indianlegal_llm.config import _parse_top_k

    assert _parse_top_k(None, 3) == 3
    assert _parse_top_k("", 3) == 3
    assert _parse_top_k("  ", 3) == 3
    assert _parse_top_k("5", 3) == 5
    for bad in ("abc", "0", "-2", "1.5"):
        with pytest.raises(ValueError):
            _parse_top_k(bad, 3)


def test_pipeline_logs_provenance_manifest(pipeline):
    """Every ingested doc is logged with url/court/date/license (CLAUDE.md §3)."""
    assert pipeline.manifest
    for entry in pipeline.manifest:
        for field in ("url", "court", "date", "license"):
            assert entry.get(field), f"manifest entry missing {field}: {entry}"


def test_answer_schema_invariant():
    """refused implies no citations; grounded implies citations."""
    refused = Answer(question="q", text="no", citations=[], refused=True)
    assert not refused.is_grounded
    grounded = Answer(
        question="q",
        text="yes [x]",
        citations=[],  # empty -> not grounded even if not refused
        refused=False,
    )
    assert not grounded.is_grounded
