# Roadmap

The journey is **stub → real**, one interface at a time, always keeping the
green-build gate (CLI runs, eval green, pytest passes) and the trust property.

## Milestone 0 — Walking skeleton ✅ (this repo)

- All interfaces (ABCs) + stdlib stubs in place.
- Pipeline runs end-to-end with zero dependencies.
- Eval: `citation_accuracy=1.0`, `refusal_accuracy=1.0`, `hallucinations=0`.

## Milestone 1 — Real ingestion (Data workstream)

- Replace `StubIngestor` with streaming ingestors for AWS Open Data (Indian
  SC/HC judgments on S3), India Code, and Indian Kanoon.
- Emit a persisted ingestion manifest (`url, court, date, license`).
- Runs **in the cloud**; corpora never hit a laptop (CLAUDE.md §5).

## Milestone 2 — Real processing

- Replace `StubProcessor` with sentence/layout-aware chunking, dedup, and
  citation-span detection. Preserve provenance per chunk.

## Milestone 3 — Real retrieval (RAG workstream)

- Add a real `BaseEmbedder` (e.g. an open-weights embedding model) and a vector
  `BaseRetriever` (FAISS/Chroma/Qdrant) selected by `VECTOR_BACKEND`.
- Keep the Answerer's refuse-unless-cited contract intact; add a relevance
  threshold so weak matches still refuse.

## Milestone 4 — Real model + MIT adapter (Model workstream)

- Wrap a confirmed Apache-2.0/MIT base model behind `BaseLLM` — default
  `microsoft/phi-4` (MIT); Gemma 4 (Apache-2.0, April 2026 release) only after
  confirming its exact repo id on the current model card (Gemma 3 and earlier do
  not qualify). Verify the license before download/fine-tune.
- Fine-tune an Indian-law LoRA/QLoRA adapter; **ship the adapter under MIT**.
- Train/infer in the cloud; only the 50–200 MB adapter is downloaded.

## Milestone 5 — Evaluation at scale (QA workstream)

- Expand cases: more landmark judgments, statute questions, adversarial citation
  attacks, and a larger out-of-domain refusal set.
- Add metrics: citation precision/recall, answer faithfulness, latency.

## Milestone 6 — Productionize surfaces

- Harden the FastAPI service (auth, rate limits, structured logging).
- Polish the Gradio demo; add source highlighting.

## Invariants across every milestone

- Code + adapters stay **MIT**; base model stays **Apache-2.0/MIT**.
- Data stays **commercially clean**; provenance is always logged.
- The Answerer **never** emits an ungrounded legal claim.
- The green-build gate stays green.
