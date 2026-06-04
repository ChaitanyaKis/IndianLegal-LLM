"""Offline tests for the Gradio demo glue (no gradio/GPU needed).

The demo module import-guards gradio, so its backend-selection and markdown
helpers are testable without gradio installed.
"""

from __future__ import annotations

from indianlegal_llm.app import demo
from indianlegal_llm.config import Settings
from indianlegal_llm.schemas import Answer, Citation


def test_demo_llm_falls_back_to_stub_without_gpu(monkeypatch):
    monkeypatch.setattr(demo, "_on_zerogpu", lambda: False)
    monkeypatch.setattr(demo, "_cuda_available", lambda: False)
    assert type(demo._demo_llm(Settings(llm="transformers"))).__name__ == "StubLLM"
    assert type(demo._demo_llm(Settings(llm="stub"))).__name__ == "StubLLM"


def test_demo_remote_unconfigured_uses_stub(monkeypatch):
    monkeypatch.delenv("REMOTE_LLM_URL", raising=False)
    assert type(demo._demo_llm(Settings(llm="remote"))).__name__ == "StubLLM"


def test_answer_to_markdown_grounded_shows_reference_and_link():
    answer = Answer(
        question="q",
        text="Privacy is protected.",
        citations=[
            Citation(
                chunk_id="puttaswamy-2017::0",
                doc_id="puttaswamy-2017",
                title="K. S. Puttaswamy v. Union of India",
                court="Supreme Court of India",
                url="https://indiankanoon.org/doc/1/",
                neutral_citation="2017 INSC 1",
                para_start=297,
                para_end=297,
            )
        ],
        refused=False,
    )
    md = demo.answer_to_markdown(answer)
    assert "K. S. Puttaswamy v. Union of India, 2017 INSC 1 ¶ 297" in md
    assert "(https://indiankanoon.org/doc/1/)" in md  # click-through link


def test_answer_to_markdown_refusal_is_graceful():
    answer = Answer(
        question="q",
        text="declined",
        citations=[],
        refused=True,
        refusal_reason="the model did not cite any retrieved source",
    )
    md = demo.answer_to_markdown(answer)
    assert "can't give a grounded answer" in md.lower()
    assert "the model did not cite any retrieved source" in md
