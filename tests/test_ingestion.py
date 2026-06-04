"""Tests for the real ingestion layer.

No real corpora are committed: fixtures are tiny synthetic JSON/HTML files that
mirror the EXACT confirmed parquet columns / page shapes. The S3 parquet read
path is exercised by building a tiny parquet in a tmp dir (pyarrow) and pointing
the ingestor at the local filesystem — never S3, never network, never committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indianlegal_llm.config import Settings
from indianlegal_llm.ingestion import get_ingestor
from indianlegal_llm.ingestion._util import html_to_text, normalize_date, safe_id
from indianlegal_llm.ingestion.aws_s3 import (
    SC_BUCKET,
    AWSHighCourtIngestor,
    AWSSupremeCourtIngestor,
)
from indianlegal_llm.ingestion.india_code import IndiaCodeIngestor
from indianlegal_llm.ingestion.indian_kanoon import IndianKanoonIngestor
from indianlegal_llm.ingestion.manifest import write_manifest
from indianlegal_llm.ingestion.stub import StubIngestor
from indianlegal_llm.processing.stub import StubProcessor
from indianlegal_llm.rag.retriever import InMemoryRetriever

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _read_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_html_to_text_strips_tags_and_scripts():
    assert html_to_text("<p>Hello <b>world</b></p><script>x()</script>") == "Hello world"
    assert html_to_text("plain text") == "plain text"
    assert html_to_text("") == ""


def test_safe_id_normalizes_to_citation_charset():
    assert safe_id("C.A. No. 123 of 2017") == "C.A.-No.-123-of-2017"
    assert safe_id("a/b\\c d") == "a-b-c-d"
    assert safe_id("   ") == ""


def test_normalize_date_converts_dmy_to_iso():
    assert normalize_date("17-10-1950") == "1950-10-17"  # real AWS format
    assert normalize_date("1-5-2017") == "2017-05-01"  # single-digit, zero-padded
    assert normalize_date("2017-08-24") == "2017-08-24"  # already ISO -> unchanged
    assert normalize_date("") == ""
    assert normalize_date("24 August, 2017") == "24 August, 2017"  # not guessed


# --------------------------------------------------------------------------- #
# Supreme Court mapping (exact columns from the Athena DDL)
# --------------------------------------------------------------------------- #
def test_sc_row_to_rawdoc_maps_fields_and_strips_html():
    rows = _load("sc_metadata_sample.json")
    doc = AWSSupremeCourtIngestor.row_to_rawdoc(rows[0])
    assert doc.doc_id == "sc-2017-INSC-1"  # neutral citation (nc_display) preferred
    assert doc.court == "Supreme Court of India"
    assert doc.date == "2017-08-24"
    assert doc.language == "en"  # single code; full list kept in metadata
    assert doc.metadata["available_languages"] == "en"
    assert doc.metadata["nc_display"] == "2017 INSC 1"
    assert doc.metadata["cnr"] == "SCIN01-000001-2012"
    assert doc.license == "government-work"
    assert doc.url == f"s3://{SC_BUCKET}/data/tar/year=2017/english/doc1.html"
    assert "Article 21" in doc.text and "track" not in doc.text  # script stripped


def test_sc_language_is_single_code_even_when_source_lists_many():
    row = {
        "case_id": "X1",
        "nc_display": "",
        "decision_date": "01-02-2020",
        "available_languages": "ENG,HIN,PUN",
        "raw_html": "<p>text</p>",
    }
    doc = AWSSupremeCourtIngestor.row_to_rawdoc(row)
    assert doc.language == "en"
    assert doc.metadata["available_languages"] == "ENG,HIN,PUN"
    assert doc.date == "2020-02-01"  # single-digit DMY normalized + zero-padded


def test_sc_party_title_fallback_is_not_corrupted():
    # 'Vodafone' starts with 'v'; the old str.strip(' v.') truncated it.
    row = {"case_id": "Z9", "petitioner": "Vodafone", "respondent": "Union of India"}
    doc = AWSSupremeCourtIngestor.row_to_rawdoc(row)
    assert doc.title == "Vodafone v. Union of India"


def test_sc_row_sanitizes_messy_case_id_and_falls_back_to_description():
    rows = _load("sc_metadata_sample.json")
    doc = AWSSupremeCourtIngestor.row_to_rawdoc(rows[1])
    assert doc.doc_id == "sc-C.A.-No.-123-of-2017"  # spaces -> '-'
    assert doc.text == "fallback description"  # no raw_html
    assert doc.court == "Supreme Court of India"  # empty court defaulted
    assert doc.url == f"s3://{SC_BUCKET}/metadata/parquet"  # no path


def test_sc_row_uses_cnr_when_case_id_missing_and_builds_title():
    rows = _load("sc_metadata_sample.json")
    doc = AWSSupremeCourtIngestor.row_to_rawdoc(rows[2])
    assert doc.doc_id == "sc-SCIN01-000999-2017"
    assert doc.title == "State v. Accused"


def test_sc_row_without_any_id_is_skipped():
    rows = _load("sc_metadata_sample.json")
    assert AWSSupremeCourtIngestor.row_to_rawdoc(rows[3]) is None


# --------------------------------------------------------------------------- #
# High Court mapping
# --------------------------------------------------------------------------- #
def test_hc_row_to_rawdoc_uses_pdf_link_as_url():
    rows = _load("hc_metadata_sample.json")
    doc = AWSHighCourtIngestor.row_to_rawdoc(rows[0])
    assert doc.doc_id == "hc-MHHC01-000123-2020"
    assert doc.url == "https://example-hc.gov.in/judgments/abc.pdf"
    assert doc.court == "Bombay High Court"
    assert doc.date == "2020-06-15"
    assert doc.license == "government-work"


def test_hc_row_without_cnr_is_skipped():
    rows = _load("hc_metadata_sample.json")
    assert AWSHighCourtIngestor.row_to_rawdoc(rows[1]) is None


# --------------------------------------------------------------------------- #
# Real parquet read path (local tmp parquet, no S3, no commit)
# --------------------------------------------------------------------------- #
def _write_local_sc_parquet(root: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = _load("sc_metadata_sample.json")
    for row in rows:
        row.pop("year", None)  # year comes from the hive partition path
    part = root / "year=2017"
    part.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(part / "metadata.parquet"))


def test_sc_ingestor_streams_local_parquet(tmp_path):
    _write_local_sc_parquet(tmp_path)
    docs = list(AWSSupremeCourtIngestor(limit=10, root=str(tmp_path)).fetch())
    assert len(docs) == 3  # 4 rows, 1 has no id and is skipped
    assert {d.doc_id for d in docs} == {
        "sc-2017-INSC-1",  # nc_display
        "sc-C.A.-No.-123-of-2017",  # case_id fallback (no nc_display/cnr)
        "sc-SCIN01-000999-2017",  # cnr fallback (no nc_display/case_id)
    }
    # year came from the partition path
    assert docs[0].metadata["year"] == "2017"


def test_sc_ingestor_respects_limit(tmp_path):
    _write_local_sc_parquet(tmp_path)
    docs = list(AWSSupremeCourtIngestor(limit=2, root=str(tmp_path)).fetch())
    assert len(docs) == 2


def test_real_doc_ids_are_citation_safe(tmp_path):
    """Chunks from real (messy) ids must pass the retriever's id validation."""
    _write_local_sc_parquet(tmp_path)
    docs = list(AWSSupremeCourtIngestor(limit=10, root=str(tmp_path)).fetch())
    processor = StubProcessor()
    chunks = [c for d in docs for c in processor.process(d)]
    InMemoryRetriever().add(chunks)  # must not raise on invalid/duplicate ids


# --------------------------------------------------------------------------- #
# Web ingestor mappers (pure; fixtures, no network)
# --------------------------------------------------------------------------- #
def test_indian_kanoon_mapper():
    doc = IndianKanoonIngestor.doc_to_rawdoc("91938676", _read_text("indian_kanoon_sample.html"))
    assert doc.doc_id == "ik-91938676"
    assert doc.court == "Supreme Court of India"
    assert doc.url == "https://indiankanoon.org/doc/91938676/"
    assert doc.title.startswith("K. S. Puttaswamy")
    assert "fundamental right" in doc.text


def test_india_code_mapper():
    url = "https://www.indiacode.nic.in/handle/123456789/2263"
    doc = IndiaCodeIngestor.page_to_rawdoc(url, _read_text("india_code_sample.html"))
    assert doc.doc_id == "ic-2263"
    assert doc.title == "The Indian Penal Code, 1860"
    assert doc.date == "1860"
    assert doc.url == url
    assert doc.license == "government-work"


def test_india_code_requires_urls():
    with pytest.raises(ValueError):
        list(IndiaCodeIngestor(limit=5).fetch())  # no urls provided


def test_india_code_rejects_non_india_code_host():
    # Jurisdiction/provenance guard: only indiacode.nic.in may be stamped.
    ingestor = IndiaCodeIngestor(limit=5, urls=["https://evil.example.com/act/1"])
    with pytest.raises(ValueError):
        list(ingestor.fetch())


# --------------------------------------------------------------------------- #
# Registry + manifest
# --------------------------------------------------------------------------- #
def test_get_ingestor_resolves_sources_and_rejects_unknown():
    assert isinstance(get_ingestor("sc", limit=5), AWSSupremeCourtIngestor)
    assert isinstance(get_ingestor("aws-hc", limit=5), AWSHighCourtIngestor)
    assert isinstance(get_ingestor("indian-kanoon", limit=5), IndianKanoonIngestor)
    assert get_ingestor("stub").source_name.startswith("stub")
    with pytest.raises(ValueError):
        get_ingestor("scc-online")  # forbidden source is not wired in


def test_write_manifest_records_provenance(tmp_path):
    docs = list(StubIngestor().fetch())
    path = tmp_path / "data" / "source_manifest.jsonl"
    count = write_manifest(docs, str(path))
    assert count == 2
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert set(entry) == {"doc_id", "url", "court", "date", "language", "license"}
    assert entry["language"] == "en"


# --------------------------------------------------------------------------- #
# build_pipeline fallback (keeps the offline skeleton green)
# --------------------------------------------------------------------------- #
def test_build_pipeline_falls_back_to_stub_on_ingestor_failure(monkeypatch, capsys):
    from indianlegal_llm import pipeline as pl
    from indianlegal_llm.ingestion.base import BaseIngestor

    class FailingIngestor(BaseIngestor):
        source_name = "failing-real"

        def fetch(self):
            raise ImportError("s3fs not installed (simulated)")
            yield  # pragma: no cover - never reached

    monkeypatch.setattr(pl, "get_ingestor", lambda *a, **k: FailingIngestor())
    pipe = pl.build_pipeline(Settings(ingestor="aws-sc"))

    assert pipe.source.startswith("stub")  # fell back
    assert pipe.num_chunks > 0
    assert pipe.answer("Is privacy a fundamental right in India?").is_grounded
    assert "falling back to StubIngestor" in capsys.readouterr().err
