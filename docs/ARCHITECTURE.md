# Architecture

IndianLegal-LLM is a **walking skeleton**: every component is an abstract
interface (ABC) with a pure-stdlib stub. The whole pipeline runs end-to-end with
zero external dependencies, and each stub is replaced later behind the same
interface.

## Data flow

```
 BaseIngestor.fetch()        -> Iterator[RawDoc]
   BaseProcessor.process()   -> list[Chunk]
     BaseRetriever.add()      (index the chunks)
     BaseRetriever.retrieve() -> list[RetrievedChunk]      (per query, top_k)
       citation.build_user_prompt(question, retrieved)
         BaseLLM.generate(system, user) -> str
           Answerer: extract cited ids -> keep only retrieved ids -> Answer | refuse
```

The contracts that travel between layers live in
[`indianlegal_llm/schemas.py`](../indianlegal_llm/schemas.py):

`RawDoc` → `Chunk` → `RetrievedChunk` → `Citation` → `Answer`.

## Components and interfaces

| Layer       | Module                          | Interface                          | Stub |
|-------------|---------------------------------|------------------------------------|------|
| Ingestion   | `ingestion/`                    | `BaseIngestor.fetch()`             | `StubIngestor` — 2 real SC judgments (Puttaswamy, Kesavananda) |
| Processing  | `processing/`                   | `BaseProcessor.process()`          | `StubProcessor` — naive overlapping word chunker |
| Embedding   | `rag/base.py`, `rag/embedder.py`| `BaseEmbedder.embed()`             | `StubEmbedder` — deterministic hashing vector |
| Retrieval   | `rag/base.py`, `rag/retriever.py`| `BaseRetriever.add()/retrieve()`  | `InMemoryRetriever` — stopword-filtered token overlap |
| Citations   | `rag/citation.py`               | (functions)                        | `SYSTEM_PROMPT`, `build_user_prompt`, `extract_cited_ids`, `to_citation` |
| Model       | `model/`                        | `BaseLLM.generate()`               | `StubLLM` — deterministic, cites the first source |
| Answering   | `rag/answerer.py`               | `Answerer.answer()`                | enforces the trust property |
| Wiring      | `pipeline.py`                   | `build_pipeline()`                 | the single wiring point |
| Eval        | `evaluation/`                   | `EvalCase`, `harness.run()`        | 3 cases; the green-build gate |
| Surfaces    | `app/`                          | —                                  | `cli.py` (stdlib), `api.py` (FastAPI, guarded), `demo.py` (Gradio, guarded) |

## The single wiring point

`build_pipeline()` in [`pipeline.py`](../indianlegal_llm/pipeline.py) is the only
place implementations are assembled. Every surface — CLI, API, demo, eval — calls
it. To swap a stub for a real implementation, pass it to `build_pipeline()` (or
change the default there); callers are unaffected because the `Base*` interfaces
are unchanged.

```python
from indianlegal_llm.pipeline import build_pipeline
from my_pkg import RealRetriever

pipeline = build_pipeline(retriever=RealRetriever())   # same interface, real impl
```

## The trust property (CLAUDE.md §4)

The Answerer is the safety boundary:

1. `retrieve(question, top_k)` applies a **relevance gate**: a chunk is returned
   only if it shares at least `min_overlap` (default 2) *distinct* content tokens
   with the query. An out-of-domain question that shares at most an incidental
   token or two yields **zero** retrieved chunks.
2. The LLM is prompted to cite retrieved ids in `[brackets]` or refuse.
3. `extract_cited_ids` parses the reply; the Answerer **keeps only ids that were
   actually retrieved**, dropping anything fabricated.
4. If no valid citation remains, the Answerer returns a refusal.
5. `chunk_id`s are validated on `InMemoryRetriever.add()` — non-empty, citation
   charset only, and unique — so a citation can never map to the wrong source or
   silently become un-citable.

Therefore a non-refused `Answer` always carries at least one citation to a
retrieved source (the *attribution* guarantee). This is covered by
`tests/test_pipeline.py` (`test_answerer_drops_fabricated_citations`,
`test_every_citation_is_a_retrieved_chunk`, `test_non_refused_answer_always_has_a_citation`,
`test_incidental_overlap_out_of_domain_is_refused`).

### Two honest limitations of the *stub*

- **Lexical relevance is coarse.** A question that shares *several* tokens with
  the corpus (e.g. "privacy under the US/German constitution") can still retrieve
  and be answered, because token overlap cannot reason about jurisdiction. The
  semantic retriever (Milestone 3) closes this; the gate above only removes the
  single-incidental-token failure class.
- **Attribution is not faithfulness.** The Answerer checks that a citation exists
  and was retrieved; it does not verify the model's prose against the cited text.
  The `StubLLM` is faithful by construction (it echoes the source); a real model
  must add a faithfulness step (Milestone 4).

## Why stdlib-only for the skeleton

The skeleton must always import and run so the green-build gate is meaningful on
any machine. Optional dependencies (FastAPI, Gradio, ML stacks) live behind
extras in `pyproject.toml` and are import-guarded, never required by the core.
