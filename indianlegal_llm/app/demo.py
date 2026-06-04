"""Gradio demo (optional dependency, import-guarded) — free HF ZeroGPU ready.

Importing this module never fails: if Gradio is not installed, ``build_demo()``
raises a clear error. Install with::

    pip install -e .[demo]      # local (falls back to the StubLLM on a laptop)

Run with::

    python -m indianlegal_llm.app.demo

ZeroGPU: on a free Hugging Face Space, only the LLM generation runs on the
dynamically-allocated GPU (decorated with ``@spaces.GPU``); retrieval and the
citation guard run on CPU. The base model + LoRA adapter are pulled HF-Hub ->
Space (cloud-to-cloud) the first time generation runs inside the GPU context —
never to a laptop (CLAUDE.md §5). Set ``LLM=remote`` to call a hosted endpoint
instead (for when ZeroGPU quota / cold-start isn't suitable).
"""

from __future__ import annotations

import importlib.util
import sys

from ..config import Settings
from ..model.base import BaseLLM
from ..model.registry import get_llm
from ..model.stub import StubLLM
from ..pipeline import build_pipeline
from ..schemas import Answer

try:  # optional dependency — guarded so the package imports with stdlib only
    import gradio as gr

    _HAS_GRADIO = True
except ImportError:  # pragma: no cover - exercised only without gradio installed
    _HAS_GRADIO = False

_EXAMPLES = [
    "Is privacy a fundamental right in India?",
    "What is the basic structure doctrine of the Indian Constitution?",
    "What is the capital of France?",
]


def _on_zerogpu() -> bool:
    """True on a Hugging Face ZeroGPU Space (the `spaces` package is present)."""
    return importlib.util.find_spec("spaces") is not None


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch import edge cases
        return False


def _gpu_decorator():
    """Return ``spaces.GPU`` on a ZeroGPU Space, else a no-op decorator."""
    if _on_zerogpu():
        import spaces  # type: ignore

        return spaces.GPU
    def _identity(fn):
        return fn

    return _identity


def _demo_llm(settings: Settings) -> BaseLLM:
    """Choose the demo's LLM, deferring GPU work to the decorated generation call.

    - ``stub``: offline, deterministic (local default fallback).
    - ``remote``: hosted OpenAI-compatible endpoint (no GPU needed).
    - ``transformers``: real model. The model is NOT loaded at startup (ZeroGPU has
      no CUDA until inside ``@spaces.GPU``); generation is wrapped so the load +
      forward happen inside the allocated-GPU context. On a CPU-only laptop we fall
      back to the stub so the app still runs.
    """
    backend = (settings.llm or "stub").strip().lower()
    if backend == "stub":
        return StubLLM()

    if backend in ("remote", "http", "endpoint"):
        llm = get_llm("remote")
        try:
            llm.ensure_loaded()  # validates the endpoint is configured (no network)
            return llm
        except Exception as exc:
            print(f"[indianlegal_llm] demo: remote LLM unavailable ({exc}); using stub.", file=sys.stderr)
            return StubLLM()

    # Real (transformers) backend: only viable on a GPU host / ZeroGPU Space.
    if not (_on_zerogpu() or _cuda_available()):
        print("[indianlegal_llm] demo: no GPU/ZeroGPU; using StubLLM for local dev.", file=sys.stderr)
        return StubLLM()

    llm = get_llm(settings.llm, base_model=settings.base_model, adapter=settings.adapter)
    gpu = _gpu_decorator()
    underlying_generate = llm.generate  # bound method; loads lazily on first call

    @gpu
    def gpu_generate(system: str, user: str) -> str:
        # Runs inside the allocated GPU context on ZeroGPU; the model loads here.
        return underlying_generate(system, user)

    llm.generate = gpu_generate  # only generation touches the GPU
    return llm


def answer_to_markdown(answer: Answer) -> str:
    """Render an Answer as chat markdown: prominent citation references + links."""
    if answer.refused:
        reason = answer.refusal_reason or "no grounded source"
        return (
            "**⚠️ I can't give a grounded answer.**\n\n"
            f"{answer.text}\n\n_Reason: {reason}._"
        )
    lines = [answer.text, "", "**Sources**"]
    for c in answer.citations:
        # The full reference — "<title>, <neutral citation> ¶ N" — is the headline,
        # with a click-through to the source url.
        headline = f"[{c.reference}]({c.url})" if c.url else c.reference
        lines.append(f"- **{headline}**  \n  `{c.chunk_id}` · {c.court}")
    return "\n".join(lines)


def build_demo():
    """Build the Gradio chat interface. Raises if Gradio is not installed."""
    if not _HAS_GRADIO:
        raise RuntimeError(
            "Gradio is not installed. Install the demo extra: pip install -e .[demo]"
        )

    settings = Settings.from_env()
    pipeline = build_pipeline(settings, llm=_demo_llm(settings))

    def respond(message: str, history: object) -> str:
        if not message or not message.strip():
            return "Please enter a question about Indian law."
        # Retrieval + citation guard run here on CPU; only llm.generate uses the GPU.
        return answer_to_markdown(pipeline.answer(message))

    return gr.ChatInterface(
        fn=respond,
        title="IndianLegal-LLM",
        description=(
            "Citation-grounded answers for **Indian law** (Supreme Court, High "
            "Courts, statutes). Every claim is backed by a verbatim quote from a "
            "retrieved source — or the assistant refuses. Each answer shows the "
            "full citation (case, neutral citation, paragraph pinpoint) and links "
            "to the source."
        ),
        examples=_EXAMPLES,
    )


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
