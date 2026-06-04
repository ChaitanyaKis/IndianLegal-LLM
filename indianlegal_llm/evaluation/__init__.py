"""Evaluation layer: the green-build gate (CLAUDE.md §6).

`harness.run()` builds the pipeline via `build_pipeline()` and scores it against
the cases in `cases.py`, reporting citation_accuracy, refusal_accuracy, and
hallucinations.
"""

from .schema import EvalCase

__all__ = ["EvalCase"]
