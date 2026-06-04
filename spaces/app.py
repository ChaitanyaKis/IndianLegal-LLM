"""Hugging Face Space entrypoint (app_file in README front-matter).

Builds the Gradio demo from the installed ``indianlegal_llm`` package. Backend
(ZeroGPU transformers vs RemoteLLM vs stub) is chosen from Space Variables; see
this directory's README.md.
"""

from indianlegal_llm.app.demo import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
