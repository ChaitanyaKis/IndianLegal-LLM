"""Zero-dependency command-line interface.

    python -m indianlegal_llm.app.cli "Is privacy a fundamental right in India?"

Pure standard library: it imports `build_pipeline()` and prints either a cited
answer or a refusal.
"""

from __future__ import annotations

import sys

from .._io import enable_utf8_output
from ..pipeline import build_pipeline
from ..schemas import Answer

_USAGE = 'usage: python -m indianlegal_llm.app.cli "your question"'


def format_answer(answer: Answer) -> str:
    """Render an Answer for the terminal (UTF-8; see _io.enable_utf8_output)."""
    lines: list[str] = []
    if answer.refused:
        lines.append("[REFUSED]")
        lines.append(answer.text)
        return "\n".join(lines)

    lines.append(answer.text)
    lines.append("")
    lines.append("Citations:")
    for c in answer.citations:
        lines.append(f"  - [{c.chunk_id}] {c.title} ({c.court})")
        lines.append(f"      {c.url}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(_USAGE, file=sys.stderr)
        return 2
    if args[0] in ("-h", "--help"):
        print(_USAGE)
        return 0

    question = args[0]
    pipeline = build_pipeline()
    answer = pipeline.answer(question)

    print(f"Q: {question}")
    print()
    print(format_answer(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
