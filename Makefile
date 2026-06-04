# IndianLegal-LLM — developer entry points.
# Uses `python` by default; override with `make PY=python3 run`.
PY ?= python
Q ?= Is privacy a fundamental right in India?

.PHONY: help run eval test ci gate eval-quality lint install dev-install api demo

help:
	@echo "Targets:"
	@echo "  run          - run the CLI (override the question with: make run Q='...')"
	@echo "  eval         - run the deterministic eval gate (must stay green)"
	@echo "  ci           - the full blocking gate: pytest + deterministic eval gate"
	@echo "  eval-quality - run the NON-blocking GPU quality eval (real model/retriever)"
	@echo "  test         - run pytest"
	@echo "  lint         - run ruff (requires the dev extra)"
	@echo "  install      - editable install of the package"
	@echo "  api          - run the FastAPI app (requires the api extra)"
	@echo "  demo         - run the Gradio demo (requires the demo extra)"

run:
	$(PY) -m indianlegal_llm.app.cli "$(Q)"

eval:
	$(PY) -m indianlegal_llm.evaluation.harness

# The same blocking gate CI runs: unit tests (incl. citation-guard invariants)
# THEN the deterministic eval gate (hallucinations=0, citation/refusal/hit-rate
# thresholds, guard self-check). Stub-pinned, offline, no GPU.
ci: test gate

gate:
	$(PY) -m indianlegal_llm.evaluation.harness

eval-quality:
	$(PY) -m indianlegal_llm.evaluation.quality

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check indianlegal_llm tests

install:
	$(PY) -m pip install -e .

dev-install:
	$(PY) -m pip install -e .[dev]

api:
	$(PY) -m uvicorn indianlegal_llm.app.api:app --reload

demo:
	$(PY) -m indianlegal_llm.app.demo
