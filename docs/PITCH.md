# IndianLegal-LLM — demo + traction summary

_For AWS Activate / NVIDIA Inception applications. One page._

## One-liner

An open-source (MIT), **citation-grounded** legal assistant for **Indian law** that
**refuses unless it can quote a retrieved authority** — with paragraph pinpoints
and answers in Indian languages.

## The problem

General LLMs hallucinate case law and citations — unacceptable in legal practice.
Indian legal AI is also English-first, while most citizens and many practitioners
work in Hindi, Telugu, Tamil, Malayalam, and other languages.

## The trust architecture (the moat)

Every answer must pass a **citation guard** before it reaches a user:
1. **Retrieved-set guard** — the model may only cite chunk ids that were actually
   retrieved; fabricated ids are dropped.
2. **Quote-grounding guard** — each quoted proposition must appear **verbatim** in
   the cited source (Unicode-normalized; negation/reorder/paraphrase/translation
   all fail); otherwise the answer is refused.
3. **Metadata-only citations** — case name, neutral citation, and paragraph
   pinpoint are built from trusted source metadata, never from model text, so a
   fabricated authority is **structurally impossible**.

Result: **no ungrounded legal claim, ever** — verified, not promised. A 26-attack
adversarial red-team is encoded as regression tests.

## Differentiation: Indic, trustworthy

A Hindi/Telugu question retrieves over the English corpus (multilingual-e5),
explains **in the user's language**, but keeps the **quote verbatim in English**
(the language of the authority) so the guard still verifies it — exactly how
lawyers work: quote the authority, explain in the vernacular.

## Metrics (deterministic gate, every PR)

`hallucinations = 0`, `citation_accuracy = 1.0`, `refusal_accuracy = 1.0`,
`retrieval_hit_rate = 1.0`, all guard invariants hold — enforced by CI as a
**blocking merge gate**. A separate GPU quality tier reports citation accuracy,
retrieval hit-rate, and LegalBench/LawBench-style task metrics.

## Why it's cheap to run (the ask is small)

- **Data:** AWS Open Data (Indian Supreme Court + 25 High Courts), processed
  **in the cloud** — only code goes up, only a 50–200 MB adapter comes down.
- **Model:** Phi-4 (MIT) zero-shot or a **QLoRA adapter on Qwen3-4B (Apache-2.0)**,
  trained on a **free T4**, served 4-bit.
- **Demo:** free Hugging Face **ZeroGPU** Space (dynamically-allocated GPU); a
  RemoteLLM/endpoint fallback for scale.

GPU credits (NVIDIA) accelerate fine-tuning at scale and lower-latency serving;
AWS credits cover cloud corpus processing + an inference endpoint.

## Links

- Repo: https://github.com/ChaitanyaKis/IndianLegal-LLM
- Live demo (ZeroGPU Space): _TODO: add Space URL after deploy (see `spaces/`)._
- Architecture: [docs/ARCHITECTURE.md](ARCHITECTURE.md) · Roadmap:
  [docs/ROADMAP.md](ROADMAP.md)
