# CLAUDE.md — IndianLegal-LLM

This file defines the **LOCKED constraints** for the IndianLegal-LLM project. They are
non-negotiable. Every change to this repository — by a human or by an AI assistant — MUST
obey them. If a request conflicts with a constraint below, refuse the request and cite the
constraint, or propose an alternative that stays within bounds.

---

## 1. Jurisdiction (LOCKED)

- **Indian law only.** Sources are limited to:
  - Supreme Court of India judgments
  - High Court judgments
  - Statutes / bare acts (e.g. via India Code)
- No foreign case law, no foreign statutes. Foreign questions must be **refused**, not answered.

## 2. Licensing (LOCKED)

- **Code + fine-tuned adapters ship under MIT.** See `LICENSE`.
- **Base model selection (precise):** the base model must be a **confirmed Apache-2.0 or MIT
  model, verified against its CURRENT Hugging Face model card before use.**
  - **Default = `microsoft/phi-4` (MIT).**
  - **Alternative = Gemma 4 (Apache-2.0 as of its April 2026 release** — NOTE: **Gemma 3 and
    earlier use Google's custom *Gemma Terms* and do NOT qualify**); if used, **confirm the exact
    repo id and parameter size on the model card, and do NOT hardcode an unverified
    `gemma-4-4b` string.**
  - ❌ **NEVER fine-tune a Llama-licensed or Gemma-custom-licensed base** for the redistributable
    model.
- Adapters (LoRA / QLoRA) we train and ship are MIT. The base model is downloaded by the user
  under its own (Apache-2.0/MIT) license; we never redistribute the base weights ourselves.

## 3. Data provenance (LOCKED)

- Data MUST be **commercially clean**. Allowed sources:
  - **AWS Open Data** — Indian Supreme Court + High Court judgments on S3.
  - **India Code** — statutes / bare acts.
  - **Indian Kanoon** — government works / public records.
- ❌ **NEVER use SCC Online, Manupatra, or ILDC** (non-commercial / restricted licensing).
- Every ingested source MUST be logged in an **ingestion manifest** with at least:
  `url, court, date, license`. See `RawDoc` and the ingestion layer.

## 4. Trust property (LOCKED, NON-NEGOTIABLE)

- The **Answerer refuses unless the model cites a retrieved source.**
- A citation is only valid if its `chunk_id` matches a chunk that was actually retrieved for
  that query. Citations to non-retrieved ids are dropped.
- **No ungrounded legal claims, ever.** If, after filtering, zero valid citations remain, the
  Answerer returns a refusal — it never emits an answer "from its own knowledge".

## 5. Cost / bandwidth (LOCKED)

- **Free-tier first.** Assume no paid infra.
- **Process corpora IN THE CLOUD; never download corpora to a laptop.**
  - Only **code / configs go up**.
  - Only a **50–200 MB adapter comes down**.
  - **Avoid pushing >2 GB through notebook I/O.**

## 6. Green-build invariant (LOCKED)

After **any** change:

1. The CLI MUST run end-to-end:
   `python -m indianlegal_llm.app.cli "Is privacy a fundamental right in India?"`
2. The evaluation harness MUST stay green:
   `python -m indianlegal_llm.evaluation.harness`
   (citation_accuracy = 1.0, refusal_accuracy = 1.0, hallucinations = 0)
3. `pytest` MUST pass.

The skeleton runs on the **Python standard library only** (zero external dependencies).
Real implementations are added later **behind the same ABC interfaces** — never by changing
the contracts in a way that breaks the walking skeleton.

---

## Architecture in one line

`ingestion → processing → rag (embed/retrieve) → model (LLM) → answerer (citation-grounded)`,
wired in exactly one place: `indianlegal_llm/pipeline.py :: build_pipeline()`. The CLI, API,
demo, and evaluation harness all call `build_pipeline()` — they never wire components directly.

## How to extend (the only sanctioned pattern)

1. Pick a `Base*` ABC (e.g. `BaseLLM`, `BaseRetriever`).
2. Write a real implementation that satisfies the same interface.
3. Swap it in inside `build_pipeline()`.
4. Re-run the three green-build checks above.

Never widen, rename, or weaken an interface to make an implementation fit. Adapt the
implementation, not the contract.
