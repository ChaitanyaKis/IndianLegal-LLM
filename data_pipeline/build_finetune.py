"""Build the REAL citation-grounded fine-tune dataset (M-FT1).

Extractive-first, zero-fabrication mining from actual SC judgments (the processed
PDF text). Every example EITHER cites a real chunk/paragraph with a VERBATIM,
guard-passing quote OR is a refusal hard-negative — no uncited legal claim.

Reuses the SHIPPED chunking (``indianlegal_llm.processing.StubProcessor``) and the
citation guard (``indianlegal_llm.rag.citation``), so chunk-ids / paragraph refs
match what retrieval serves (train == inference). Output matches the QLoRA chat
schema (messages = system/user/assistant).

Run (build-time, on the local corpus):
    python -m data_pipeline.build_finetune --years 2018-2026 --out data/finetune
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from indianlegal_llm.processing.stub import StubProcessor
from indianlegal_llm.rag.citation import assess_citations, build_user_prompt
from indianlegal_llm.rag.citation import SYSTEM_PROMPT
from indianlegal_llm.schemas import Chunk, RawDoc, RetrievedChunk

from .corpus import load_processed

REFUSAL_ANSWER = (
    "I can't find support for this question in the provided sources, so I am not "
    "able to answer it."
)

_HOLDING = re.compile(
    r"(held that|we hold|in our (considered )?opinion|we are of the (view|opinion)|"
    r"it is (well[- ]?)?settled|this court has held|we are of the considered view)",
    re.I,
)
_ISSUE = re.compile(
    r"(question for consideration|question that arises|issue (that arises|is|for)|"
    r"the point for (determination|consideration)|whether )",
    re.I,
)
_SECTION = re.compile(r"(?:[Ss]ection|[Ss]\.)\s*\d+[A-Za-z]?|[Aa]rticle\s+\d+")
_ACT = re.compile(r"([A-Z][A-Za-z&. ]{3,60}?Act,?\s*\d{4})")
_SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9\"(])")

_MAX_QUOTE_WORDS = 60
_MIN_QUOTE_WORDS = 6
_MAX_POS_PER_DOC = 4


@dataclass
class Example:
    type: str
    doc_id: str  # the source/context judgment (split is by this)
    question: str
    answer: str
    context_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    messages: list[dict]
    split: str = ""


@dataclass
class Summary:
    by_type: dict = field(default_factory=dict)
    positives: int = 0
    refusals: int = 0
    dropped_guard: int = 0
    dropped_dup: int = 0
    train: int = 0
    dev: int = 0
    docs: int = 0
    dev_docs: int = 0
    answer_words: list = field(default_factory=list)


# --- text helpers ---------------------------------------------------------- #
def _short_title(title: str) -> str:
    t = re.sub(r"\s+", " ", title).strip()
    t = re.sub(r"\s+(versus|vs\.?|v/s)\s+", " v. ", t, flags=re.I)
    return (t[:90] + "...") if len(t) > 93 else t


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _quote(chunk_text: str, signal: re.Pattern | None) -> str | None:
    """A verbatim quote from the chunk (a sentence matching ``signal``), capped.

    When a long sentence is truncated to the word cap, back off to the last clause
    boundary (. ; :) within the window so the quote does not end mid-clause — it
    stays a verbatim prefix of the sentence (hence a substring of the chunk).
    """
    norm_chunk = re.sub(r"\s+", " ", chunk_text)
    candidates = (
        [s for s in _sentences(chunk_text) if signal.search(s)]
        if signal is not None
        else _sentences(chunk_text)
    )
    for s in candidates:
        words = s.split()
        if len(words) < _MIN_QUOTE_WORDS:
            continue
        quote = " ".join(words[:_MAX_QUOTE_WORDS])
        if len(words) > _MAX_QUOTE_WORDS:
            cut = max(quote.rfind(". "), quote.rfind("; "), quote.rfind(": "))
            if cut != -1 and len(quote[: cut + 1].split()) >= _MIN_QUOTE_WORDS:
                quote = quote[: cut + 1]
        quote = quote.strip()
        if quote and quote in norm_chunk:  # verbatim (modulo whitespace) substring
            return quote
    return None


def _pinpoint(chunk: Chunk) -> str:
    ps, pe = chunk.metadata.get("para_start"), chunk.metadata.get("para_end")
    if ps is None:
        return ""
    return f" ¶ {ps}" if (pe is None or pe == ps) else f" ¶ {ps}-{pe}"


def _act_ref(doc_text: str) -> str:
    m = _ACT.search(doc_text)
    return m.group(1).strip() if m else ""


# Generic words that are NOT distinctive of a judgment's subject-matter; excluded
# from topic tokens so the refusal absence-check spends its budget only on terms
# that actually identify the foreign question's topic.
_LEGAL_STOPWORDS = {
    "state", "union", "india", "court", "supreme", "others", "anr", "another",
    "limited", "company", "private", "commissioner", "criminal", "procedure",
    "appeal", "appellant", "respondent", "versus", "through", "department",
    "government", "municipal", "corporation", "board", "authority", "officer",
    "district", "singh", "kumar", "delhi", "bench", "civil", "matter", "case",
    "petition", "application", "order", "judgment", "republic", "association",
}


def _doc_topic_tokens(title: str, text: str) -> set[str]:
    """Distinctive (non-generic) topic words for refusal non-overlap checks."""
    toks = set()
    act = _act_ref(text)
    if act:
        toks.update(w.lower() for w in act.split() if len(w) > 4)
    toks.update(w.lower().strip(".,&'") for w in title.split() if len(w) > 4)
    return {t for t in toks if t and t not in _LEGAL_STOPWORDS}


# --- mining ---------------------------------------------------------------- #
def _make_messages(question: str, context: list[Chunk], answer: str) -> list[dict]:
    retrieved = [RetrievedChunk(chunk=c, score=1.0) for c in context]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, retrieved)},
        {"role": "assistant", "content": answer},
    ]


def _positive(
    etype: str, question: str, cited: Chunk, context: list[Chunk], quote: str,
    doc_id: str, extra_quote: tuple[Chunk, str] | None = None,
) -> Example | None:
    answer = f'The Court observed: "{quote}" [{cited.chunk_id}]{_pinpoint(cited)}.'
    cited_ids = [cited.chunk_id]
    if extra_quote is not None:
        c2, q2 = extra_quote
        answer += f' It further held: "{q2}" [{c2.chunk_id}]{_pinpoint(c2)}.'
        cited_ids.append(c2.chunk_id)
    ctx: list[Chunk] = []
    seen_ids: set[str] = set()
    for c in context + [cited] + ([extra_quote[0]] if extra_quote else []):
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            ctx.append(c)
    by_id = {c.chunk_id: c for c in ctx}
    valid, reason = assess_citations(answer, by_id)
    if reason is not None or not set(cited_ids).issubset(set(valid)):
        return None  # guard drop
    return Example(
        type=etype, doc_id=doc_id, question=question, answer=answer,
        context_chunk_ids=[c.chunk_id for c in ctx], cited_chunk_ids=cited_ids,
        messages=_make_messages(question, ctx, answer),
    )


def mine_positives(doc_id: str, title: str, chunks: list[Chunk], doc_text: str) -> tuple[list[Example], int]:
    short, act = _short_title(title), _act_ref(doc_text)
    examples: list[Example] = []
    dropped = 0
    body = [c for c in chunks if c.metadata.get("para_start") is not None] or chunks

    def add(ex: Example | None):
        nonlocal dropped
        if ex is None:
            dropped += 1
        elif len(examples) < _MAX_POS_PER_DOC:
            examples.append(ex)

    # issue-framing
    for c in body:
        q = _quote(c.text, _ISSUE)
        if q:
            add(_positive("issue", f"What was the question for consideration in {short}?",
                          c, _neighbors(chunks, c), q, doc_id))
            break
    # statutory interpretation
    for c in body:
        if _SECTION.search(c.text):
            q = _quote(c.text, _SECTION)
            if q:
                sec = _SECTION.search(c.text).group(0)
                of_act = f" of the {act}" if act else ""
                add(_positive("statutory", f"How did the Supreme Court interpret {sec}{of_act} in {short}?",
                              c, _neighbors(chunks, c), q, doc_id))
                break
    # holding / ratio
    holdings = [(c, _quote(c.text, _HOLDING)) for c in body]
    holdings = [(c, q) for c, q in holdings if q]
    if holdings:
        c, q = holdings[0]
        about = f" on the {act}" if act else ""
        add(_positive("holding", f"What did the Supreme Court hold{about} in {short}?",
                      c, _neighbors(chunks, c), q, doc_id))
    # multi-paragraph synthesis (2 distinct holding chunks)
    if len(holdings) >= 2 and len(examples) < _MAX_POS_PER_DOC:
        (c1, q1), (c2, q2) = holdings[0], holdings[1]
        if c1.chunk_id != c2.chunk_id:
            add(_positive("multi", f"Summarise the Supreme Court's reasoning in {short}.",
                          c1, _neighbors(chunks, c1), q1, doc_id, extra_quote=(c2, q2)))
    return examples, dropped


def _neighbors(chunks: list[Chunk], c: Chunk, n: int = 2) -> list[Chunk]:
    """Up to n sibling chunks (distractors) from the same doc, excluding c."""
    return [x for x in chunks if x.chunk_id != c.chunk_id][:n]


def mine_refusal(ctx_doc_id: str, ctx_chunks: list[Chunk], foreign_q: str) -> Example:
    ctx = ctx_chunks[:3]
    return Example(
        type="refusal", doc_id=ctx_doc_id, question=foreign_q, answer=REFUSAL_ANSWER,
        context_chunk_ids=[c.chunk_id for c in ctx], cited_chunk_ids=[],
        messages=_make_messages(foreign_q, ctx, REFUSAL_ANSWER),
    )


# --- build ------------------------------------------------------------------ #
def build(records: Iterable[dict], *, dev_fraction: float = 0.08, refusal_target: float = 0.30) -> tuple[list[Example], list[Example], Summary]:
    processor = StubProcessor()
    docs = []  # (doc_id, title, chunks, text, topic_tokens, a positive question)
    summary = Summary()
    seen_questions: set[str] = set()

    positives: list[Example] = []
    for rec in records:
        doc = RawDoc(
            doc_id=rec["doc_id"], title=rec.get("title", ""),
            court=rec.get("court", "Supreme Court of India"), date=rec.get("date", ""),
            url=rec.get("url", ""), license=rec.get("license", "government-work"),
            text=rec["text"], language="en",
            metadata={"nc_display": rec.get("nc_display", "")},
        )
        chunks = processor.process(doc)
        if not chunks:
            continue
        exs, dropped = mine_positives(doc.doc_id, doc.title, chunks, doc.text)
        summary.dropped_guard += dropped
        kept = []
        for ex in exs:
            key = re.sub(r"\s+", " ", ex.question.lower()).strip()
            if key in seen_questions:
                summary.dropped_dup += 1
                continue
            seen_questions.add(key)
            kept.append(ex)
        positives.extend(kept)
        docs.append({
            "doc_id": doc.doc_id, "chunks": chunks, "text": doc.text,
            "topic": _doc_topic_tokens(doc.title, doc.text),
            "question": kept[0].question if kept else None,
        })

    # Refusal hard-negatives: ask doc F's question against doc D's context, where
    # F's distinctive topic tokens are ABSENT from D (a true unsupported negative).
    n_pos = len(positives)
    target_refusals = int(round(refusal_target / (1 - refusal_target) * n_pos)) if n_pos else 0
    refusals: list[Example] = []
    with_q = [d for d in docs if d["question"]]
    used: set[tuple[str, str]] = set()
    per_doc: dict[str, int] = {}
    # Word set of exactly the chunks the model is SHOWN for each doc (ctx_chunks[:3]).
    # A refusal is a genuine hard-negative iff NONE of the foreign question's
    # distinctive topic words appear in that shown context.
    shown_words = {
        d["doc_id"]: set(re.findall(r"[a-z0-9]+", " ".join(c.text for c in d["chunks"][:3]).lower()))
        for d in docs
    }
    # Multiple rounds: pair each doc with different foreign topics (each genuinely
    # unsupported by the SHOWN context) until the refusal ratio target is met.
    for rnd in range(1, 7):
        if len(refusals) >= target_refusals or not with_q:
            break
        progressed = False
        for i, d in enumerate(docs):
            if len(refusals) >= target_refusals:
                break
            if per_doc.get(d["doc_id"], 0) >= 3:
                continue
            f = with_q[(i + rnd) % len(with_q)]
            if f["doc_id"] == d["doc_id"]:
                continue
            ftoks = f["topic"]
            if not ftoks or any(t in shown_words[d["doc_id"]] for t in ftoks):
                continue  # foreign topic word present in shown context -> not a hard negative
            key = (d["doc_id"], f["question"])
            if key in used:
                continue
            used.add(key)
            per_doc[d["doc_id"]] = per_doc.get(d["doc_id"], 0) + 1
            refusals.append(mine_refusal(d["doc_id"], d["chunks"], f["question"]))
            progressed = True
        if not progressed:
            break

    # Split BY judgment (no doc leakage). Deterministic: every Nth doc -> dev.
    all_doc_ids = [d["doc_id"] for d in docs]
    step = max(1, int(round(1 / dev_fraction))) if dev_fraction > 0 else 0
    dev_ids = {doc_id for k, doc_id in enumerate(all_doc_ids) if step and k % step == 0}

    train, dev = [], []
    for ex in positives + refusals:
        ex.split = "dev" if ex.doc_id in dev_ids else "train"
        (dev if ex.split == "dev" else train).append(ex)
        summary.by_type[ex.type] = summary.by_type.get(ex.type, 0) + 1
        summary.answer_words.append(len(ex.answer.split()))

    summary.positives = len(positives)
    summary.refusals = len(refusals)
    summary.train = len(train)
    summary.dev = len(dev)
    summary.docs = len(docs)
    summary.dev_docs = len(dev_ids)
    return train, dev, summary


def _write(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps({
                "type": ex.type, "doc_id": ex.doc_id, "split": ex.split,
                "question": ex.question, "answer": ex.answer,
                "context_chunk_ids": ex.context_chunk_ids,
                "cited_chunk_ids": ex.cited_chunk_ids, "messages": ex.messages,
            }, ensure_ascii=False) + "\n")


def _print_summary(summary: Summary) -> None:
    total = summary.train + summary.dev
    refusal_pct = (summary.refusals / total * 100) if total else 0.0
    considered = total + summary.dropped_guard + summary.dropped_dup
    guard_pct = (summary.dropped_guard / considered * 100) if considered else 0.0
    words = sorted(summary.answer_words)

    def pct(p):
        return words[min(len(words) - 1, int(p * len(words)))] if words else 0

    print("\n=== M-FT1 fine-tune dataset summary ===")
    print(f"docs processed: {summary.docs} (dev docs: {summary.dev_docs})")
    print(f"examples: {total}  (train={summary.train}, dev={summary.dev})")
    print(f"by type: {dict(sorted(summary.by_type.items()))}")
    print(f"positives={summary.positives}  refusals={summary.refusals}  "
          f"refusal_share={refusal_pct:.1f}% (target 25-35%)")
    print(f"dropped by guard: {summary.dropped_guard} ({guard_pct:.1f}% of considered)  "
          f"dropped dup: {summary.dropped_dup}")
    print(f"answer length (words): p50={pct(0.5)} p90={pct(0.9)} max={words[-1] if words else 0}")


def main(argv: list[str] | None = None) -> int:
    from indianlegal_llm._io import enable_utf8_output

    enable_utf8_output()
    parser = argparse.ArgumentParser(prog="python -m data_pipeline.build_finetune")
    parser.add_argument("--processed", default="data/sc/processed", help="processed JSONL dir")
    parser.add_argument("--out", default="data/finetune", help="output dir for train/dev jsonl")
    parser.add_argument("--dev-fraction", type=float, default=0.08)
    parser.add_argument("--refusal-target", type=float, default=0.30)
    args = parser.parse_args([] if argv is None else argv)

    records = list(load_processed(Path(args.processed)))
    if not records:
        print(f"no processed records in {args.processed}; run data_pipeline.corpus first.", file=sys.stderr)
        return 2
    train, dev, summary = build(
        records, dev_fraction=args.dev_fraction, refusal_target=args.refusal_target
    )
    out = Path(args.out)
    _write(train, out / "train.jsonl")
    _write(dev, out / "dev.jsonl")
    _print_summary(summary)
    print(f"wrote {out/'train.jsonl'} and {out/'dev.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
