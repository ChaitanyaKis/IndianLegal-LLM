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


def _parse_positive_int(raw: str | None, default: int, name: str = "value") -> int:
    """Parse a positive int from env defensively: clear error on garbage."""
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _parse_top_k(raw: str | None, default: int) -> int:
    """Parse TOP_K defensively: clear error on garbage, must be positive."""
    return _parse_positive_int(raw, default, name="TOP_K")


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _parse_bool(raw: str | None, default: bool, name: str = "value") -> bool:
    """Parse a boolean from env: 1/true/yes/on vs 0/false/no/off (case-insensitive)."""
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean (1/0/true/false), got {raw!r}")


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
    ingestor:
        Which ingestion source to use. One of "stub", "aws-sc", "aws-hc",
        "india-code", "indian-kanoon". The real sources need the `ingestion`
        extra; `build_pipeline()` falls back to "stub" if they are unavailable so
        the zero-dependency skeleton always runs (CLAUDE.md §6).
    ingest_limit:
        Max documents to pull from a real source (the ingestion `--limit`).
    manifest_path:
        Where the ingestion manifest is written (under the gitignored data/).
    ingestor_strict:
        If True, a missing [ingestion] extra / network / credential is a HARD
        error instead of falling back to the stub. Default False (graceful
        fallback with a stderr warning). Env: INGESTOR_STRICT=1.
    llm:
        Which LLM backend to serve. "transformers" (real, loads ``base_model``
        4-bit on a CUDA GPU) or "stub" (offline, deterministic). The real backend
        needs the `model` extra + a GPU; `build_pipeline()` falls back to "stub"
        if it is unavailable so the skeleton always runs offline (CLAUDE.md §6).
        Env: LLM. The eval harness is always pinned to the stub.
    """

    base_model: str = "microsoft/phi-4"
    embedding_model: str = "stub-token-overlap"
    vector_backend: str = "memory"
    top_k: int = 3
    ingestor: str = "aws-sc"
    ingest_limit: int = 200
    manifest_path: str = "data/source_manifest.jsonl"
    ingestor_strict: bool = False
    llm: str = "transformers"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build Settings from environment variables, falling back to defaults."""
        return cls(
            base_model=os.getenv("BASE_MODEL", cls.base_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            vector_backend=os.getenv("VECTOR_BACKEND", cls.vector_backend),
            top_k=_parse_top_k(os.getenv("TOP_K"), cls.top_k),
            ingestor=os.getenv("INGESTOR", cls.ingestor),
            ingest_limit=_parse_top_k(os.getenv("INGEST_LIMIT"), cls.ingest_limit),
            manifest_path=os.getenv("SOURCE_MANIFEST", cls.manifest_path),
            ingestor_strict=_parse_bool(
                os.getenv("INGESTOR_STRICT"), cls.ingestor_strict, name="INGESTOR_STRICT"
            ),
            llm=os.getenv("LLM", cls.llm),
        )

    def base_model_is_license_clean(self) -> bool:
        """True if the configured base model is on the MIT/Apache-2.0 allowlist."""
        return self.base_model in LICENSE_CLEAN_BASE_MODELS
