---
name: Feature request
about: Propose an enhancement (new ingestor, retriever, model, eval case, language)
title: "[feat] "
labels: enhancement
---

**The need**
What problem does this solve? Who benefits (e.g. a specific Indian jurisdiction /
language / source)?

**Proposed approach**
Which `Base*` interface does it extend (`BaseIngestor`, `BaseRetriever`,
`BaseEmbedder`, `BaseLLM`)? How does it wire into `build_pipeline()`?

**Constraints (CLAUDE.md)**
- Jurisdiction: Indian law only.
- Licensing: base model Apache-2.0/MIT; adapters MIT.
- Data: commercially clean (AWS Open Data / India Code / Indian Kanoon) — never
  SCC Online / Manupatra / ILDC.
- Trust property + green-build must hold.

**Golden-set / eval**
If it changes behavior, which eval cases (deterministic and/or quality) cover it?
