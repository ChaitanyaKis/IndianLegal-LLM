"""Offline tests for the build-time data_pipeline (no corpus / no PDF needed).

Asserts the M-FT1 HARD RULES on synthetic judgment text:
- every train answer is EITHER a guard-passing cited answer OR a refusal,
- zero source-judgment overlap between train and dev,
- the SCR cleaner + extractor behave, and the local-sc ingestor reads processed text.
"""

from __future__ import annotations

import json

from data_pipeline import build_finetune as bf
from data_pipeline.pdf_text import clean_scr_text, extract_pdf_text
from indianlegal_llm.processing.stub import StubProcessor
from indianlegal_llm.rag.citation import REASON_NO_CITATION, assess_citations
from indianlegal_llm.schemas import RawDoc

# Synthetic judgments with numbered paragraphs + holding/issue/section signals and
# DISTINCT topics (so refusal cross-pairing finds genuinely unsupported questions).
# Distinct topics with DISTINCTIVE act names (so each doc's topic tokens are
# non-empty and pairwise disjoint — exercising the refusal absence-check honestly).
_TOPICS = [
    ("lease", "Transfer of Property Act, 1872", "Section 105",
     "whether the agreement was a lease", "the agreement was not a lease"),
    ("arbitration", "Arbitration and Conciliation Act, 1996", "Section 9",
     "whether interim relief is available", "interim relief is available"),
    ("privacy", "Constitution of India", "Article 21",
     "whether privacy is protected", "privacy is a fundamental right"),
    ("narcotics", "Narcotic Drugs and Psychotropic Substances Act, 1985", "Section 37",
     "whether bail ought to be granted", "bail ought to be granted on conditions"),
    ("taxation", "Income Tax Act, 1961", "Section 147",
     "whether reassessment was valid", "the reassessment was without jurisdiction"),
    ("environment", "Environment Protection Act, 1986", "Section 15",
     "whether the clearance was lawful", "the environmental clearance was unlawful"),
]


def _synthetic_records() -> list[dict]:
    records = []
    for i, (topic, act, sec, issue, holding) in enumerate(_TOPICS):
        text = (
            f"IN THE SUPREME COURT OF INDIA. Appeal No. {i} of 2020. {topic.title()} matter.\n\n"
            f"1. This appeal concerns the {act} and turns on {sec}.\n\n"
            f"2. The question for consideration is {issue} under {sec} of the {act}.\n\n"
            f"3. On a true interpretation of {sec} of the {act}, the provision must be "
            f"read in the context of its object and purpose in {topic} disputes.\n\n"
            f"4. We hold that {holding}, and we are of the view that the appeal is "
            f"disposed of accordingly in light of {sec}.\n"
        )
        records.append({
            "doc_id": f"sc-2020TEST{i}", "title": f"APPELLANT {i} v. RESPONDENT {i}",
            "court": "Supreme Court of India", "date": "2020-01-01",
            "url": f"s3://x/{i}", "license": "government-work",
            "nc_display": f"2020TEST{i}", "cnr": "", "citation": "", "year": "2020",
            "text": text,
        })
    return records


def _doc_chunk_index(records):
    proc = StubProcessor()
    out = {}
    for rec in records:
        doc = RawDoc(
            doc_id=rec["doc_id"], title=rec["title"], court=rec["court"],
            date=rec["date"], url=rec["url"], license=rec["license"],
            text=rec["text"], language="en", metadata={"nc_display": rec["nc_display"]},
        )
        out[rec["doc_id"]] = {c.chunk_id: c for c in proc.process(doc)}
    return out


def test_extractor_quarantines_non_pdf_and_cleaner_strips_scr_noise():
    assert extract_pdf_text(b"not a real pdf at all") is None  # no text layer
    dirty = "A\nB\n791\nSUPREME  COURT  REPORTS\n1. The Court held that X.\n2. And further Y.\n[2020] 10 S.C.R."
    cleaned = clean_scr_text(dirty)
    assert "SUPREME" not in cleaned
    assert "\nA\n" not in f"\n{cleaned}\n" and not cleaned.startswith("A")
    assert "1. The Court held that X." in cleaned
    assert "2. And further Y." in cleaned


def test_build_produces_grounded_or_refusal_only_and_no_doc_leakage():
    records = _synthetic_records()
    chunk_index = _doc_chunk_index(records)
    train, dev, summary = bf.build(records, dev_fraction=0.2, refusal_target=0.30)

    assert train and dev
    assert summary.refusals > 0 and summary.positives > 0

    # HARD RULE 1+3: every answer is a refusal OR a guard-passing cited answer.
    for ex in train + dev:
        if ex.type == "refusal":
            assert ex.answer == bf.REFUSAL_ANSWER
            assert ex.cited_chunk_ids == []
        else:
            by_id = {cid: chunk_index[ex.doc_id][cid] for cid in ex.context_chunk_ids}
            valid, reason = assess_citations(ex.answer, by_id)
            assert reason is None, f"{ex.type} answer not grounded: {ex.answer}"
            assert set(ex.cited_chunk_ids).issubset(set(valid))

    # No uncited legal claim: a non-refusal must carry a citation.
    assert all(ex.cited_chunk_ids for ex in train + dev if ex.type != "refusal")

    # Split BY judgment: zero doc overlap.
    train_docs = {ex.doc_id for ex in train}
    dev_docs = {ex.doc_id for ex in dev}
    assert train_docs.isdisjoint(dev_docs)


def test_refusal_share_in_target_band():
    train, dev, summary = bf.build(_synthetic_records(), dev_fraction=0.2, refusal_target=0.30)
    total = len(train) + len(dev)
    share = summary.refusals / total
    assert 0.20 <= share <= 0.40  # builder targets ~30%; allow band for small sets


def test_refusal_answer_is_uncited():
    # A refusal must never be mistaken for a grounded claim by the guard.
    valid, reason = assess_citations(bf.REFUSAL_ANSWER, {"sc-x::0": None})
    assert valid == [] and reason == REASON_NO_CITATION


def test_local_sc_ingestor_reads_processed_jsonl(tmp_path):
    from indianlegal_llm.ingestion.local_sc import LocalSCIngestor

    processed = tmp_path / "year=2020.jsonl"
    rec = _synthetic_records()[0]
    processed.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    docs = list(LocalSCIngestor(processed_dir=str(tmp_path), limit=10).fetch())
    assert len(docs) == 1
    assert docs[0].doc_id == "sc-2020TEST0"
    assert docs[0].metadata["nc_display"] == "2020TEST0"
    assert "Transfer of Property Act" in docs[0].text


def test_local_sc_ingestor_errors_when_no_processed_corpus(tmp_path):
    import pytest

    from indianlegal_llm.ingestion.local_sc import LocalSCIngestor

    with pytest.raises(FileNotFoundError):
        list(LocalSCIngestor(processed_dir=str(tmp_path)).fetch())
