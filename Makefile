# IndianLegal-LLM — developer entry points.
# Uses `python` by default; override with `make PY=python3 run`.
PY ?= python
Q ?= Is privacy a fundamental right in India?

.PHONY: help run eval test lint install dev-install api demo

help:
	@echo "Targets:"
	@echo "  run     - run the CLI (override the question with: make run Q='...')"
	@echo "  eval    - run the evaluation harness (must stay green)"
	@echo "  test    - run pytest"
	@echo "  lint    - run ruff (requires the dev extra)"
	@echo "  install - editable install of the package"
	@echo "  api     - run the FastAPI app (requires the api extra)"
	@echo "  demo    - run the Gradio demo (requires the demo extra)"

run:
	$(PY) -m indianlegal_llm.app.cli "$(Q)"

eval:
	$(PY) -m indianlegal_llm.evaluation.harness

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
