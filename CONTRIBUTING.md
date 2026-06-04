# Contributing to IndianLegal-LLM

Thanks for helping build a trustworthy, open legal-AI framework for Indian law.

## Read this first

`CLAUDE.md` at the repo root defines **LOCKED constraints**. They are
non-negotiable and apply to every contribution (human or AI-assisted):

1. **Jurisdiction:** Indian law only (Supreme Court, High Courts, statutes/bare acts).
2. **Licensing:** code + adapters are MIT; the base model must be Apache-2.0 or MIT.
3. **Data:** commercially clean sources only (AWS Open Data, India Code, Indian
   Kanoon). Never SCC Online, Manupatra, or ILDC. Log every source.
4. **Trust property:** the Answerer refuses unless it can cite a retrieved source.
5. **Cost/bandwidth:** process corpora in the cloud; only code/configs up, only a
   50–200 MB adapter down.
6. **Green build:** the CLI runs, the eval harness is green, and pytest passes.

A PR that violates a LOCKED constraint will not be merged.

## The green-build gate

Before opening a PR, all three MUST hold:

```bash
python -m indianlegal_llm.app.cli "Is privacy a fundamental right in India?"
python -m indianlegal_llm.evaluation.harness   # citation=1.0 refusal=1.0 halluc=0
python -m pytest -q
```

The skeleton itself runs on the **standard library only**. Don't add a hard
runtime dependency to the core package; put optional dependencies behind extras
in `pyproject.toml` and guard their imports (see `app/api.py`, `app/demo.py`).

## The only sanctioned way to add a real implementation

The project is a *walking skeleton*: every component is an ABC + a stub.

1. Pick a `Base*` interface (e.g. `BaseLLM`, `BaseRetriever`, `BaseIngestor`).
2. Write a real class that satisfies the **same** interface — do not change,
   widen, or weaken the interface to fit your implementation.
3. Wire it in **only** inside `indianlegal_llm/pipeline.py :: build_pipeline()`.
   Nothing else constructs components directly.
4. Re-run the green-build gate above.

If you genuinely need to evolve a contract, that's a design discussion: open an
issue first and update `docs/ARCHITECTURE.md` together with the change.

## Data contributions

Every ingested document must carry full provenance (`url`, `court`, `date`,
`license`) and be logged in the ingestion manifest. PRs adding data must update
`docs/DATA_SOURCES.md` and justify the license. See `CODEOWNERS` for reviewers.

## Style

- Keep the core import-light and stdlib-only.
- `ruff` for linting (`make lint`), `pytest` for tests (`make test`).
- Match the surrounding code; document the *why*, not the *what*.
