# IndianLegal-LLM

**An open-source, citation-grounded LLM framework for Indian law.**

IndianLegal-LLM answers questions about **Indian law** (Supreme Court, High
Courts, statutes/bare acts) and **refuses to make any claim it cannot cite to a
retrieved source**. It is MIT-licensed and designed to be redistributed cleanly.

> This repository is a **walking skeleton**: every component is defined by an
> abstract interface (ABC) and ships with a pure-standard-library stub. The full
> pipeline runs end-to-end with **zero external dependencies**. Real
> implementations replace the stubs later, behind the same interfaces.

See [CLAUDE.md](CLAUDE.md) for the **LOCKED constraints** that govern every change.

---

## Quickstart (zero dependencies)

Requires only Python 3.10+. No `pip install` needed to run the skeleton.

```bash
# Ask an in-domain question -> get a cited answer
python -m indianlegal_llm.app.cli "Is privacy a fundamental right in India?"

# Ask an out-of-domain question -> get a refusal (no ungrounded claim)
python -m indianlegal_llm.app.cli "What is the capital of France?"

# Run the evaluation harness (the green-build gate)
python -m indianlegal_llm.evaluation.harness
```

Expected harness output:

```
citation_accuracy=1.0
refusal_accuracy=1.0
hallucinations=0
BUILD: GREEN
```

With `make`:

```bash
make run                 # CLI on the default question
make run Q="What is the basic structure doctrine?"
make eval                # evaluation harness
make test                # pytest
```

---

## The trust property (non-negotiable)

The **Answerer refuses unless the model cites a retrieved source.** Concretely:

```
retrieve -> build prompt -> LLM -> keep only citations whose id was retrieved -> refuse if none
```

- A citation is valid **only** if its `chunk_id` was actually retrieved for that
  query. Citations to anything else are dropped.
- If no valid citation survives, the Answerer returns a **refusal**.
- There is **no path** by which an ungrounded legal claim is returned.

This is enforced in
[`indianlegal_llm/rag/answerer.py`](indianlegal_llm/rag/answerer.py) and verified
by the test suite and the evaluation harness.

---

## Architecture

```
ingestion -> processing -> rag (retrieve*) -> model (LLM) -> answerer
            (RawDoc)        (Chunk)            (text)         (Answer + Citations)
```

\* The skeleton's retrieval is **lexical** (`InMemoryRetriever`, token overlap).
`StubEmbedder` ships as a placeholder for the future vector retriever but is
**not wired into `build_pipeline()`** yet — see Milestone 3 in
[docs/ROADMAP.md](docs/ROADMAP.md).

Wired in exactly one place — `build_pipeline()` in
[`indianlegal_llm/pipeline.py`](indianlegal_llm/pipeline.py). The CLI, API, demo,
and evaluation harness all call it; none of them construct components directly.

| Layer        | Interface (ABC)                     | Skeleton stub            |
|--------------|-------------------------------------|--------------------------|
| Ingestion    | `BaseIngestor.fetch()`              | `StubIngestor` (offline) + real `aws-sc`/`aws-hc`/`india-code`/`indian-kanoon` |
| Processing   | `BaseProcessor.process()`           | `StubProcessor` (naive chunker)      |
| Embedding    | `BaseEmbedder.embed()`              | `StubEmbedder` (hashing) — *present but not wired into the skeleton* |
| Retrieval    | `BaseRetriever.add()/retrieve()`    | `InMemoryRetriever` (lexical overlap)|
| Model        | `BaseLLM.generate()`                | `StubLLM` (cites first source)       |
| Answering    | `Answerer.answer()`                 | enforces the trust property          |

Each stub is replaced by a real implementation **behind the same interface**.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Licensing & data (clean by construction)

- **Code + fine-tuned adapters: MIT** (see [LICENSE](LICENSE)).
- **Base model: a confirmed Apache-2.0 or MIT model**, verified against its current
  Hugging Face model card before use. Default is `microsoft/phi-4` (MIT); Gemma 4
  (Apache-2.0 as of its April 2026 release) is an alternative *once its exact repo id
  is confirmed* (Gemma 3 and earlier use Google's custom Gemma Terms and do **not**
  qualify). We never fine-tune or redistribute a Llama/Gemma-custom-licensed base.
- **Data: commercially clean** — AWS Open Data (Indian SC/HC judgments), India
  Code, Indian Kanoon. **Never** SCC Online, Manupatra, or ILDC. Every source is
  logged with `url, court, date, license`. See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

---

## Real ingestion (AWS Open Data + web sources)

The skeleton answers from the offline stub corpus by default for zero-dep,
deterministic runs. To pull real Indian judgments/statutes:

```bash
pip install -e .[ingestion]

# Stream a small sample from the AWS Open Data Supreme Court bucket (ap-south-1,
# public). Writes a provenance manifest to data/source_manifest.jsonl (gitignored).
python -m indianlegal_llm.ingestion --source sc --limit 5
python -m indianlegal_llm.ingestion --source hc --limit 20            # High Courts
python -m indianlegal_llm.ingestion --source indian-kanoon --doc-ids 91938676,257876
```

Sources: `aws-sc`, `aws-hc` (parquet streamed from S3), `india-code`,
`indian-kanoon` (web) — all behind the one `BaseIngestor` interface. The corpus is
streamed and never downloaded whole to disk (CLAUDE.md §5). Set `INGESTOR=aws-sc`
(default) or `INGESTOR=stub` to choose what `build_pipeline()` indexes; if a real
source isn't available it falls back to the stub so the skeleton always runs. See
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Optional surfaces

```bash
pip install -e .[api]    # then: uvicorn indianlegal_llm.app.api:app --reload
pip install -e .[demo]   # then: python -m indianlegal_llm.app.demo
```

All optional surfaces and ingestors import-guard their dependencies, so the
package always imports with the standard library alone.

---

## Project docs

- [CLAUDE.md](CLAUDE.md) — LOCKED constraints (read first)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, contracts, data flow
- [docs/ROLES.md](docs/ROLES.md) — workstreams and ownership
- [docs/ROADMAP.md](docs/ROADMAP.md) — stub → real, milestone by milestone
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — allowed/forbidden sources & provenance
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add a real implementation

## License

MIT. See [LICENSE](LICENSE).
