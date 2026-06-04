# Roles & workstreams

Each workstream owns one or more `Base*` interfaces and the job of replacing the
stub with a real implementation **without changing the contract**. Wiring always
happens in `pipeline.py :: build_pipeline()`.

## Data / Ingestion

- **Owns:** `BaseIngestor`, `StubIngestor`, `docs/DATA_SOURCES.md`.
- **Goal:** stream Indian SC/HC judgments and statutes from license-clean sources
  (AWS Open Data S3, India Code, Indian Kanoon) as `RawDoc`s with full provenance.
- **Hard rules:** only commercially clean sources; never SCC Online / Manupatra /
  ILDC; log `url, court, date, license` for every document (CLAUDE.md §3).
- **Bandwidth:** process corpora **in the cloud**; never download corpora locally
  (CLAUDE.md §5).

## Processing

- **Owns:** `BaseProcessor`, `StubProcessor`.
- **Goal:** turn raw judgments into clean, retrievable `Chunk`s (sentence/layout
  aware, dedup, headnote/paragraph detection). Preserve provenance onto chunks.

## RAG (embedding + retrieval + citations)

- **Owns:** `BaseEmbedder`, `BaseRetriever`, `InMemoryRetriever`, `StubEmbedder`,
  `rag/citation.py`, `Answerer`.
- **Goal:** replace lexical overlap with real embeddings + a vector store
  (`VECTOR_BACKEND`), while keeping the Answerer's refuse-unless-cited guarantee.
- **Hard rule:** never weaken the trust property (CLAUDE.md §4).

## Model / fine-tuning

- **Owns:** `BaseLLM`, `StubLLM`.
- **Goal:** wrap a real base model and an MIT-licensed adapter behind `BaseLLM`.
- **Hard rule:** the base model MUST be Apache-2.0 or MIT (e.g. Gemma 4 4B, Phi-4);
  the redistributable artifact is the MIT adapter, **never** a Llama/Gemma-custom
  fine-tune (CLAUDE.md §2). Training/inference run in the cloud; only the
  50–200 MB adapter comes down (CLAUDE.md §5).

## Evaluation / QA

- **Owns:** `evaluation/` (`EvalCase`, `cases.py`, `harness.py`).
- **Goal:** grow the case set (more in-domain answers, more out-of-domain refusals,
  adversarial citation attacks) while keeping `citation_accuracy`,
  `refusal_accuracy`, and `hallucinations` honest. The harness is the green gate.

## Application / surfaces

- **Owns:** `app/` (`cli.py`, `api.py`, `demo.py`).
- **Goal:** keep the CLI stdlib-only; keep API/demo behind guarded optional extras.
- **Hard rule:** surfaces call `build_pipeline()` — they never wire components.

## Maintainers

- **Owns:** `CLAUDE.md`, `LICENSE`, `pipeline.py`, the trust-critical RAG paths.
- **Goal:** guard the LOCKED constraints and the green-build invariant. See
  `CODEOWNERS`.
