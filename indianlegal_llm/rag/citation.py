"""Citation plumbing: the system prompt, prompt builder, and citation guard.

This module encodes the trust property (CLAUDE.md section 4) at the prompt
boundary:

* ``SYSTEM_PROMPT`` instructs the model to cite a retrieved ``chunk_id`` or refuse.
* ``build_user_prompt`` lays out the retrieved sources with their ids.
* ``extract_cited_ids`` parses ``[chunk_id]`` markers out of the model's reply.
* ``to_citation`` turns a retrieved chunk into a :class:`Citation` (metadata only).
* ``assess_citations`` applies the retrieved-set and quote-grounding guards.

The Answerer calls ``assess_citations`` and refuses with its reason when needed.
"""

from __future__ import annotations

import re
import unicodedata

from ..schemas import Chunk, Citation, RetrievedChunk

# Returned verbatim by the Answerer whenever no valid citation survives filtering.
REFUSAL_MESSAGE = (
    "I can't provide a grounded answer to this question from the retrieved Indian "
    "legal sources. To avoid an ungrounded legal claim, I'm declining to answer."
)

# Specific refusal reasons (CLAUDE.md section 4 - refuse with a clear reason).
REASON_NO_CITATION = "the model did not cite any retrieved source"
REASON_UNGROUNDED = "the cited passage(s) could not be grounded in the retrieved sources"


def refusal_text(reason: str) -> str:
    """A refusal message that names why no grounded answer could be given."""
    return (
        "I can't provide a grounded answer from the retrieved Indian legal sources "
        f"({reason}). To avoid an ungrounded legal claim, I'm declining to answer."
    )


SYSTEM_PROMPT = (
    "You are IndianLegal-LLM, an assistant for Indian law only (Supreme Court of "
    "India, High Courts, and Indian statutes/bare acts).\n"
    "\n"
    "RULES (non-negotiable):\n"
    "1. Answer ONLY using the numbered sources provided in the user message.\n"
    "2. Every factual or legal statement you make MUST be supported by a citation "
    "to the source's identifier, written in square brackets, e.g. the bracketed "
    "id shown next to each source.\n"
    "3. Cite ONLY identifiers that appear in the provided sources. Never invent an "
    "identifier and never cite from memory.\n"
    "4. If the provided sources do not support an answer - including any non-Indian"
    "-law or out-of-domain question - you MUST refuse and cite nothing.\n"
    "5. When you put text in quotation marks, it MUST be verbatim from a cited "
    "source. Do not paraphrase inside quotes, and never state a case name or "
    "citation that is not in the cited source.\n"
    "6. No ungrounded legal claims, ever."
)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Render the question and retrieved sources into the user prompt.

    Each source is a header line ``[<chunk_id>] <court> - <title>`` followed by the
    chunk text. The id is the first bracketed token of each block so a model can
    align its citation to a real, retrieved id.
    """
    lines = [f"Question: {question}", "", "Sources:"]
    if not retrieved:
        lines.append("(none)")
    else:
        for rc in retrieved:
            c = rc.chunk
            lines.append(f"[{c.chunk_id}] {c.court} - {c.title}")
            lines.append(c.text)
            lines.append("")  # blank line separates sources
    lines.append(
        "Instructions: Answer using ONLY the sources above and cite the "
        "square-bracketed identifier of each source you rely on. If the sources "
        "do not contain the answer, refuse and cite nothing."
    )
    return "\n".join(lines)


# chunk_id charset: letters, digits, and the separators we use (._:-). No brackets.
_CITED_ID_RE = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")


def extract_cited_ids(text: str) -> list[str]:
    """Return the bracketed ids cited in ``text``, de-duplicated, in first-seen order."""
    seen: dict[str, None] = {}
    for match in _CITED_ID_RE.findall(text):
        seen.setdefault(match, None)
    return list(seen)


def to_citation(chunk: Chunk) -> Citation:
    """Build a :class:`Citation` from a chunk that was actually retrieved.

    EVERY human-readable field (title, neutral citation, pinpoint) comes from the
    retrieved chunk's trusted metadata - NEVER from the model's free text. A
    fabricated case name or neutral citation is therefore structurally impossible
    in a Citation. ``para_start``/``para_end`` enable pinpoint citation; absent/None
    when the source text was unnumbered.
    """
    return Citation(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        court=chunk.court,
        url=chunk.url,
        neutral_citation=chunk.metadata.get("nc_display", "") or "",
        para_start=chunk.metadata.get("para_start"),
        para_end=chunk.metadata.get("para_end"),
    )


# --- Quote grounding (span verification) ----------------------------------- #
# A "quoted proposition" is text the model puts in any double-quote-style glyph.
# Every quoted proposition in the answer MUST appear verbatim in some retrieved
# chunk; if any does not, the whole answer is refused (CLAUDE.md section 4). The
# guard runs on the FULL text with citation markers stripped (so a marker placed
# inside a quote cannot hide it), is glyph-agnostic and dangling-quote aware, and
# matches verbatim only (so a fabricated/inverted/reordered quote can never pass).

# Double-quote-style glyphs canonicalized to a straight quote before extraction:
# straight ", curly U+201C/U+201D, guillemets U+00AB/U+00BB and U+2039/U+203A,
# low-9 U+201E, high-reversed U+201F, double prime U+2033. Single quotes are
# excluded so apostrophes are never mistaken for quotes.
_DOUBLE_QUOTE_CHARS = frozenset(
    chr(cp)
    for cp in (0x22, 0x201C, 0x201D, 0x00AB, 0x00BB, 0x2039, 0x203A, 0x201E, 0x201F, 0x2033)
)
# Zero-width chars stripped during normalization: ZWSP, ZWNJ, ZWJ, ZWNBSP/BOM, WJ.
_ZERO_WIDTH = frozenset(chr(cp) for cp in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060))


def _normalize(text: str) -> str:
    """Aggressively normalize for matching: NFKD decomposition, strip combining
    marks and zero-width chars, lowercase, and reduce to alphanumeric tokens.

    NFKD + combining-mark stripping folds diacritic/homoglyph smuggling (e.g.
    a combined acute on 'not' -> 'not'); reducing punctuation to spaces makes
    verbatim matching robust to trailing periods, quotes, and bracket residue -
    WITHOUT admitting any token change (paraphrase, negation, reorder, and
    edge-token swaps all alter the token stream and so fail the match).
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if ch not in _ZERO_WIDTH)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _canonicalize_quotes(text: str) -> str:
    return "".join('"' if ch in _DOUBLE_QUOTE_CHARS else ch for ch in text)


def extract_quotes(text: str) -> list[str]:
    """Return quoted propositions, glyph-agnostic and dangling-quote aware.

    All double-quote glyphs are canonicalized to ``"``; quote characters are then
    paired in order. An unmatched (dangling) opening quote pairs with end-of-text,
    so omitting a closing quote cannot smuggle an unverified proposition.
    """
    canonical = _canonicalize_quotes(text)
    positions = [i for i, ch in enumerate(canonical) if ch == '"']
    quotes: list[str] = []
    for k in range(0, len(positions), 2):
        start = positions[k]
        end = positions[k + 1] if k + 1 < len(positions) else len(canonical)
        inner = canonical[start + 1 : end].strip()
        if len(inner) >= 3:
            quotes.append(inner)
    return quotes


def is_quote_grounded(quote: str, chunk_text: str) -> bool:
    """True only if ``quote`` appears VERBATIM (modulo normalization) in the chunk.

    The normalized quote must be a contiguous substring of the normalized chunk.
    Normalization is the only tolerance - it absorbs formatting/encoding differences
    but NOT a single changed/added/removed/reordered token, so a fabricated or
    inverted legal proposition can never be judged grounded.
    """
    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return True
    return normalized_quote in _normalize(chunk_text)


def assess_citations(
    text: str, retrieved_by_id: dict[str, Chunk]
) -> tuple[list[str], str | None]:
    """Apply both guards. Return (valid_cited_ids, refusal_reason).

    ``refusal_reason`` is None when the answer may stand; otherwise it names why:

    1. Retrieved-set guard: keep only cited ids actually retrieved. None -> refuse
       with ``REASON_NO_CITATION``.
    2. Quote-grounding guard (global): every quoted proposition in the answer must
       be grounded in SOME retrieved chunk. If any is grounded in none -> refuse
       with ``REASON_UNGROUNDED`` (a fabricated quote cannot ride a valid citation).
    """
    valid = [cid for cid in extract_cited_ids(text) if cid in retrieved_by_id]
    if not valid:
        return [], REASON_NO_CITATION

    # Strip citation markers so one placed inside a quotation cannot break it.
    marker_free = _CITED_ID_RE.sub(" ", text)
    chunks = list(retrieved_by_id.values())
    for quote in extract_quotes(marker_free):
        if not any(is_quote_grounded(quote, chunk.text) for chunk in chunks):
            return [], REASON_UNGROUNDED

    return valid, None
