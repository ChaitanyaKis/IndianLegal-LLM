"""Runtime settings, read from environment variables.

Nothing here requires external packages. Defaults are chosen so the walking
skeleton runs with zero configuration, while still pointing at license-clean
choices (see CLAUDE.md §2): the default base model is Phi-4 (MIT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Base models confirmed license-clean for MIT redistribution (CLAUDE.md §2).
# This is advisory metadata; the skeleton never downloads a base model.
#
# POLICY: only list a model here once its license is CONFIRMED Apache-2.0 or MIT
# against its CURRENT Hugging Face model card, using its EXACT repo id. Do not add
# an unverified placeholder id. Gemma 4 (Apache-2.0 as of its April 2026 release)
# may be added once you confirm the exact repo id + parameter size on the model
# card; Gemma 3 and earlier use Google's custom Gemma Terms and do NOT qualify.
LICENSE_CLEAN_BASE_MODELS = {
    "microsoft/phi-4": "MIT",  # default — genuinely MIT
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
