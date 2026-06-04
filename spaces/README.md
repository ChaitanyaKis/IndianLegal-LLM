---
title: IndianLegal-LLM
emoji: ⚖️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.11"
pinned: false
short_description: Citation-grounded answers for Indian law (ZeroGPU)
---

# IndianLegal-LLM — Hugging Face Space

A free, citation-grounded legal assistant for **Indian law**. Every claim is
backed by a verbatim quote from a retrieved source (case, neutral citation,
paragraph pinpoint) — or the assistant refuses. Retrieval and the citation guard
run on CPU; only LLM generation runs on the GPU.

This directory is the Space repo content: copy `README.md`, `app.py`, and
`requirements.txt` to a new Space (or push this `spaces/` dir as the Space root).
No weights, adapters, or corpora are committed — the Space pulls the model from
the HF Hub at runtime (cloud-to-cloud), never to a laptop (CLAUDE.md §5).

## Path A — ZeroGPU (free, dynamically-allocated GPU)

1. Create a **Gradio** Space and set its hardware to **ZeroGPU** (free, daily
   quota). `requirements.txt` includes `spaces`, `transformers`, `torch`,
   `bitsandbytes`, `peft`.
2. Set Space **Variables**:
   - `LLM=transformers`
   - `BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507`  (Apache-2.0) — or `microsoft/phi-4`
   - `LORA_ADAPTER=<your-hf-id-or-path>`  (optional; the MIT fine-tuned adapter)
3. The demo wraps generation in `@spaces.GPU`, so the model loads + runs inside
   the allocated-GPU context the first time a question is asked. The base model
   and adapter download from the HF Hub to the Space on first use.

## Path B — RemoteLLM (hosted inference endpoint)

When ZeroGPU quota / cold-start isn't suitable, point at an OpenAI-compatible
endpoint instead (no GPU on the Space). Set Space **Variables/Secrets**:
- `LLM=remote`
- `REMOTE_LLM_URL=https://<your-endpoint>/v1/chat/completions`
- `REMOTE_LLM_MODEL=<model name>`
- `REMOTE_LLM_API_KEY=<secret>`  (store as a Space *secret*)

## Local

`pip install -e .[demo]` then `python -m indianlegal_llm.app.demo`. With no GPU it
falls back to the deterministic StubLLM, so the UI still runs offline.

The citation guard (grounded-or-refuse) applies on every backend.
