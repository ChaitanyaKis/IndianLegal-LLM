<!-- Thanks for contributing to IndianLegal-LLM! -->

## What & why

<!-- Brief description of the change and the motivation. -->

## Checklist

- [ ] Obeys the LOCKED constraints in **CLAUDE.md** (jurisdiction, licensing,
      data provenance, the trust property, bandwidth, green-build).
- [ ] `make ci` is green locally (pytest + the deterministic eval gate):
      `hallucinations=0`, citation/refusal/retrieval thresholds met, guard
      invariants hold.
- [ ] New behavior added **behind an existing `Base*` interface** and wired in
      `build_pipeline()` (contracts not widened/weakened).
- [ ] **No** weights, adapters, datasets, or corpora committed
      (`git ls-files | grep -Ei 'parquet|jsonl|pdf|safetensors|adapters/'` is empty).
- [ ] Docs/golden-set updated if relevant (and lawyer-verifiable eval cases use
      `verified_by`).

## Eval impact

<!-- Paste the metrics-delta from the CI job summary, or `make eval` output. -->
