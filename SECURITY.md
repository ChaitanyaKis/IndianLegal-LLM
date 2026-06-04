# Security Policy

## Reporting a vulnerability

Please report security issues **privately**, not in public issues:

- Use GitHub's private vulnerability reporting: the repository's **Security** tab →
  **Report a vulnerability**.
- Or contact the maintainer listed in [CODEOWNERS](CODEOWNERS).

We aim to acknowledge within a few days. Please include repro steps and impact.

## Scope

This project is a **citation-grounded** research framework, not legal advice. The
core security/safety property is the **trust property** (see CLAUDE.md §4): the
Answerer refuses unless it cites a retrieved source, and a citation's
human-readable fields are built only from trusted chunk metadata — never from
model free text.

We especially welcome reports that defeat the **citation guard**, e.g.:

- a fabricated case name or neutral citation appearing in a `Citation`,
- a `[chunk_id]` that was not retrieved surviving into an answer,
- a quoted proposition that is **not** verbatim in the cited chunk being accepted
  as grounded (including cross-lingual: a translated quote must be refused),
- any non-refused answer that carries no valid citation (`hallucinations > 0`).

The deterministic eval gate (`python -m indianlegal_llm.evaluation.harness`)
encodes these as invariants; a regression there is treated as a security issue.

## Out of scope

- Output quality / legal correctness of a model's prose (this is a research
  framework; always verify against the cited primary source).
- Third-party model/base weights downloaded by the user under their own licenses.

## Supported versions

The `main` branch is supported. There are no released versions yet.
