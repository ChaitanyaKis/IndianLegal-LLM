"""data_pipeline — BUILD-TIME data preparation (NOT part of the shipped MIT path).

This package turns the on-disk SC corpus (tars of judgment PDFs) into a clean,
processed text corpus that BOTH retrieval and fine-tuning consume — so the same
chunk-ids / paragraph structure feed inference and training (train == inference).

LICENSE ISOLATION (non-negotiable, see CLAUDE.md §2):
- The shipped/inference path (`indianlegal_llm`) must NEVER import this package.
- PDF text extraction uses PERMISSIVE libraries only in the shipped sense:
  pdfminer.six (MIT) preferred, pypdf (BSD) fallback — these are normal deps.
- PyMuPDF (AGPL) is allowed ONLY here, as an explicit build-time fallback
  (`allow_agpl=True`, installed via requirements-dataprep.txt), and is never
  imported by inference/RAG/serving.
- Output is clean regardless: Indian court judgments are freely reproducible
  (Copyright Act §52(1)(q)); the only concern is keeping AGPL out of the shipped
  dependency tree.

Entry points:
- ``data_pipeline.corpus``        : tar/PDF -> cleaned processed JSONL (+ coverage).
- ``data_pipeline.build_finetune``: processed JSONL -> citation-grounded train/dev.
"""

__all__ = ["pdf_text", "corpus", "build_finetune"]
