"""PDF text extraction + SCR cleaning (build-time only).

Permissive-first extraction: pdfminer.six (MIT) -> pypdf (BSD). PyMuPDF (AGPL) is
used ONLY when ``allow_agpl=True`` (build-time data-prep extra). A PDF with no
extractable text layer (older scans) returns ``None`` so the caller can quarantine
it rather than pulling in heavier deps. ``clean_scr_text`` strips the Supreme Court
Reports margin/header noise so the cleaned text is the judgment body — this cleaned
text is what BOTH retrieval and fine-tuning chunk (train == inference).
"""

from __future__ import annotations

import io
import re

# A PDF that yields fewer than this many characters is treated as having no usable
# text layer (likely a scan) and is quarantined.
_MIN_TEXT_CHARS = 400


def _try_pdfminer(raw: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text  # MIT
    except ImportError:
        return ""
    try:
        return extract_text(io.BytesIO(raw)) or ""
    except Exception:
        return ""


def _try_pypdf(raw: bytes) -> str:
    try:
        import pypdf  # BSD
    except ImportError:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _try_fitz(raw: bytes) -> str:  # pragma: no cover - build-only AGPL fallback
    try:
        import fitz  # PyMuPDF, AGPL — build-time data-prep only
    except ImportError:
        return ""
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""


def extract_pdf_text(raw: bytes, *, allow_agpl: bool = False) -> str | None:
    """Return extracted text, or None if the PDF has no usable text layer.

    Order: pdfminer.six (MIT) -> pypdf (BSD) -> [only if allow_agpl] PyMuPDF (AGPL).
    """
    for extractor in (_try_pdfminer, _try_pypdf):
        text = extractor(raw)
        if len(text.strip()) >= _MIN_TEXT_CHARS:
            return text
    if allow_agpl:
        text = _try_fitz(raw)
        if len(text.strip()) >= _MIN_TEXT_CHARS:
            return text
    return None


# --- SCR (Supreme Court Reports) cleaning ---------------------------------- #
# Margin markers are single letters A-H on their own line; running headers repeat
# the reporter name / citation; bare page numbers appear between blocks.
_MARGIN_LETTER = re.compile(r"^[A-H]$")
_BARE_PAGE_NUM = re.compile(r"^\d{1,4}$")
_SCR_HEADER = re.compile(r"^(SUPREME COURT REPORTS|\[\d{4}\]\s*\d*\s*S\.?C\.?R\.?.*)$", re.I)


def clean_scr_text(text: str) -> str:
    """Strip SCR margin letters, running headers, and bare page numbers.

    Preserves numbered-paragraph markers (e.g. "55.") and paragraph structure so
    downstream segmentation finds the same paragraphs.
    """
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            kept.append("")
            continue
        norm = re.sub(r"\s+", " ", line)  # PDF extraction often double-spaces headers
        if _MARGIN_LETTER.match(norm) or _BARE_PAGE_NUM.match(norm) or _SCR_HEADER.match(norm):
            continue
        kept.append(norm)
    # Collapse 3+ blank lines to a single blank; trim.
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
