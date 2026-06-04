"""Gradio demo surface (optional dependency, import-guarded).

Importing this module never fails: if Gradio is not installed, ``build_demo()``
raises a clear error. Install with::

    pip install -e .[demo]

Run with::

    python -m indianlegal_llm.app.demo
"""

from __future__ import annotations

from ..pipeline import build_pipeline
from .cli import format_answer

try:  # optional dependency — guarded so the package imports with stdlib only
    import gradio as gr

    _HAS_GRADIO = True
except ImportError:  # pragma: no cover - exercised only without gradio installed
    _HAS_GRADIO = False


def build_demo():
    """Build the Gradio interface. Raises if Gradio is not installed."""
    if not _HAS_GRADIO:
        raise RuntimeError(
            "Gradio is not installed. Install the demo extra: pip install -e .[demo]"
        )

    pipeline = build_pipeline()

    def respond(question: str) -> str:
        if not question or not question.strip():
            return "Please enter a question about Indian law."
        return format_answer(pipeline.answer(question))

    return gr.Interface(
        fn=respond,
        inputs=gr.Textbox(label="Question (Indian law)", lines=2),
        outputs=gr.Textbox(label="Answer (citation-grounded, or refusal)"),
        title="IndianLegal-LLM",
        description=(
            "Citation-grounded answers for Indian law. The assistant refuses "
            "unless it can cite a retrieved source."
        ),
        examples=[
            "Is privacy a fundamental right in India?",
            "What is the basic structure doctrine of the Indian Constitution?",
            "What is the capital of France?",
        ],
    )


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
