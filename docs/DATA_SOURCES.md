# Data sources

All data in IndianLegal-LLM MUST be **commercially clean** and **Indian-law only**
(CLAUDE.md §3). Provenance (`url`, `court`, `date`, `license`) is logged for every
document in the ingestion manifest.

## ✅ Allowed sources

| Source | What | Why it's clean |
|--------|------|----------------|
| **AWS Open Data** | Indian Supreme Court + High Court judgments hosted on S3 | Open data program; processed in the cloud (no local download) |
| **India Code** | Central & state statutes / bare acts | Government of India publication |
| **Indian Kanoon** | Judgments and bare acts (government works) | Reproductions of government works / public records |

Government works (judgments, statutes) are public records; we still log their
source and a license note for traceability.

## ❌ Forbidden sources (never use)

| Source | Reason |
|--------|--------|
| **SCC Online** | Proprietary; non-commercial / restricted licensing |
| **Manupatra** | Proprietary; non-commercial / restricted licensing |
| **ILDC** (Indian Legal Documents Corpus) | Released for **non-commercial** use only |

Using any forbidden source would taint the MIT redistribution. Do not ingest,
mirror, scrape, or train on them.

## Provenance manifest

Every ingested document is logged with at least:

```json
{
  "doc_id": "puttaswamy-2017",
  "title": "K. S. Puttaswamy v. Union of India",
  "court": "Supreme Court of India",
  "date": "2017-08-24",
  "url": "https://indiankanoon.org/doc/91938676/",
  "license": "Public record (Indian Kanoon — government works)"
}
```

`build_pipeline()` collects these into `Pipeline.manifest`. The `RawDoc` schema
makes the four provenance fields mandatory, so an ingestor cannot silently drop
them.

## Skeleton data

The `StubIngestor` bakes in two landmark Supreme Court judgments as original,
factual summaries (not copies of any proprietary headnote):

- **K. S. Puttaswamy v. Union of India** (2017) — privacy as a fundamental right.
- **Kesavananda Bharati v. State of Kerala** (1973) — the basic structure doctrine.

These exist only to make the walking skeleton runnable; the real corpus arrives
via the Milestone 1 ingestors.

## Bandwidth rule (CLAUDE.md §5)

Corpora are processed **in the cloud**. Only code/configs go up; only the
50–200 MB adapter comes down. Avoid pushing >2 GB through notebook I/O.
