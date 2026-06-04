"""Runtime settings, read from environment variables.

Nothing here requires external packages. Defaults are chosen so the walking
skeleton runs with zero configuration, while still pointing at license-clean
choices (see CLAUDE.md §2): the default base model is Phi-4 (MIT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Base models that are license-clean for MIT redistribution (CLAUDE.md §2).
# This is advisory metadata; the skeleton never downloads a base model.
#
# VERIFY BEFORE USE: every entry's license string MUST be confirmed against the
# model card before that base is downloaded or fine-tuned. In particular, "Gemma
# 4 4B = Apache-2.0" is carried here from the LOCKED premise in CLAUDE.md §2, but
# real Google Gemma releases historically ship under the custom *Gemma Terms of
# Use* (non-OSI), NOT Apache-2.0 — confirm the exact variant/license, or prefer
# Phi-4 (genuinely MIT) as the redistribution-safe default. See docs/ROADMAP.md.
LICENSE_CLEAN_BASE_MODELS = {
    "microsoft/phi-4": "MIT",
    "google/gemma-4-4b": "Apache-2.0",  # <-- verify against model card (see above)
}


def _parse_top_k(raw: str | None, default: int) -> int:
    """Parse TOP_K defensively: clear error on garbage, must be positive."""
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"TOP_K must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"TOP_K must be a positive integer, got {value}")
    return value


@dataclass
class Settings:
    """Configuration for a pipeline build.

    Attributes
    ----------
    base_model:
        HF-style id of the base model. MUST be Apache-2.0 or MIT (CLAUDE.md §2).
    embedding_model:
        Id of the embedding backend. The skeleton uses "stub-token-overlap".
    vector_backend:
        Retrieval backend. The skeleton uses "memory" (InMemoryRetriever).
    top_k:
        Number of chunks to retrieve per query.
    """

    base_model: str = "microsoft/phi-4"
    embedding_model: str = "stub-token-overlap"
    vector_backend: str = "memory"
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        """Build Settings from environment variables, falling back to defaults."""
        return cls(
            base_model=os.getenv("BASE_MODEL", cls.base_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            vector_backend=os.getenv("VECTOR_BACKEND", cls.vector_backend),
            top_k=_parse_top_k(os.getenv("TOP_K"), cls.top_k),
        )

    def base_model_is_license_clean(self) -> bool:
        """True if the configured base model is on the MIT/Apache-2.0 allowlist."""
        return self.base_model in LICENSE_CLEAN_BASE_MODELS
