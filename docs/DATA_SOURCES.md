# Data sources

All data in IndianLegal-LLM MUST be **commercially clean** and **Indian-law only**
(CLAUDE.md §3). Provenance (`url`, `court`, `date`, `license`) is logged for every
document in the ingestion manifest.

## ✅ Allowed sources

| Source | What | Bucket / endpoint (confirmed) | Ingestor |
|--------|------|-------------------------------|----------|
| **AWS Open Data — Supreme Court** | SC judgments 1950–2025 | `s3://indian-supreme-court-judgments` (ap-south-1, `--no-sign-request`) | `aws-sc` |
| **AWS Open Data — High Courts** | 25 High Courts | `s3://indian-high-court-judgments` (ap-south-1, `--no-sign-request`) | `aws-hc` |
| **India Code** | Central/state statutes / bare acts | https://www.indiacode.nic.in | `india-code` |
| **Indian Kanoon** | Judgments (government works) | https://indiankanoon.org/doc/&lt;id&gt;/ | `indian-kanoon` |

Government works (judgments, statutes) are public records; we still log their
source and a license note for traceability.

### Confirmed S3 layout + parquet schema (do not guess — read from the registry)

```
# Supreme Court (partition: year)
s3://indian-supreme-court-judgments/metadata/parquet/year=YYYY/metadata.parquet
  columns: title, petitioner, respondent, description, judge, author_judge,
           citation, case_id, cnr, decision_date, disposal_nature, court,
           available_languages, raw_html, path, nc_display, scraped_at

# High Court (partitions: year, court)
s3://indian-high-court-judgments/metadata/parquet/year=YYYY/court=XXX/metadata.parquet
  columns: court_code, title, description, judge, pdf_link, cnr,
           date_of_registration, decision_date, disposal_nature, court_name
```

Mapping → `RawDoc`: `doc_id` from `nc_display` (neutral citation) → `cnr` →
`case_id`, sanitized to the citation charset; `url` from the S3 object path (SC)
or `pdf_link` (HC); `court`; `date` normalized to ISO (falls back to the `year`
partition); `language="en"` for the mapped English text, with the full
`available_languages` list kept in `metadata`; `text` from `raw_html` (SC,
HTML-stripped) or `description` (HC); `license="government-work"`.

## ❌ Forbidden sources (never use)

| Source | Reason |
|--------|--------|
| **SCC Online** | Proprietary; non-commercial / restricted licensing |
| **Manupatra** | Proprietary; non-commercial / restricted licensing |
| **ILDC** (Indian Legal Documents Corpus) | Released for **non-commercial** use only |

Using any forbidden source would taint the MIT redistribution. Do not ingest,
mirror, scrape, or train on them.

## Running real ingestion

```bash
pip install -e .[ingestion]

# Pull a small sample first (writes data/source_manifest.jsonl, gitignored)
python -m indianlegal_llm.ingestion --source sc --limit 5
python -m indianlegal_llm.ingestion --source hc --limit 20
python -m indianlegal_llm.ingestion --source indian-kanoon --doc-ids 91938676,257876
python -m indianlegal_llm.ingestion --source india-code --urls @acts.txt
```

The corpus is **streamed** from S3 (parquet read file-by-file, stopping at
`--limit`) and **never downloaded whole to local disk** (CLAUDE.md §5). Documents
are de-duplicated by `doc_id` (real corpora repeat case ids); a citation always
maps to exactly one source.

### Strict vs graceful fallback

By default, if the `[ingestion]` extra, network, or AWS credentials are missing,
the answering pipeline (`build_pipeline()`) and the ingestion CLI **fall back to
the offline stub** with a stderr warning, so the skeleton always runs. For CI /
automation where a silent fallback would hide a real problem, opt into **strict
mode** to make it a hard error instead:

```bash
INGESTOR_STRICT=1 python -m indianlegal_llm.app.cli "Is privacy a fundamental right in India?"
python -m indianlegal_llm.ingestion --source sc --limit 5 --strict   # exits non-zero on failure
```

The eval harness is always pinned to the stub, so it stays green regardless.

## Provenance manifest

Every ingested document is logged (JSONL, one line per doc):

```json
{
  "doc_id": "sc-2017-INSC-1",
  "url": "s3://indian-supreme-court-judgments/2017_10_1_…",
  "court": "Supreme Court of India",
  "date": "2017-08-24",
  "language": "ENG,HIN",
  "license": "government-work"
}
```

`build_pipeline()` also collects these into `Pipeline.manifest`. The `RawDoc`
schema makes the provenance fields mandatory, so an ingestor cannot silently drop
them. The manifest lives under the gitignored `data/` and is never committed.

## Skeleton data

The `StubIngestor` (source `stub`) bakes in two landmark Supreme Court judgments
as original, factual summaries (not copies of any proprietary headnote):

- **K. S. Puttaswamy v. Union of India** (2017) — privacy as a fundamental right.
- **Kesavananda Bharati v. State of Kerala** (1973) — the basic structure doctrine.

It runs offline with zero dependencies, so the walking skeleton and the eval
harness are always green. `build_pipeline()` falls back to it automatically when a
real source is unavailable.

## Bandwidth rule (CLAUDE.md §5)

Corpora are processed **in the cloud**. Only code/configs go up; only the
50–200 MB adapter comes down. Avoid pushing >2 GB through notebook I/O.
