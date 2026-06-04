"""Remote LLM backend: call an HTTP inference endpoint (OpenAI-compatible).

Selected with ``LLM=remote``. Useful when a local GPU isn't suitable (e.g. a
ZeroGPU Space's daily quota / cold-start): point at a hosted, OpenAI-compatible
``/v1/chat/completions`` endpoint instead. No weights are downloaded anywhere.

Config (env): ``REMOTE_LLM_URL`` (required), ``REMOTE_LLM_MODEL`` (default
"indianlegal"), ``REMOTE_LLM_API_KEY`` (optional bearer token). The HTTP client is
imported lazily, so the package still imports with the standard library alone.

The citation guard still runs on the returned text, so a remote model's output is
held to the same grounded-or-refuse contract as any other backend.
"""

from __future__ import annotations

import os

from .base import BaseLLM


class RemoteLLM(BaseLLM):
    """Calls a hosted OpenAI-compatible chat-completions endpoint."""

    model_id = "remote"

    def __init__(
        self,
        url: str | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.url = url if url is not None else os.getenv("REMOTE_LLM_URL", "")
        self.model = model or os.getenv("REMOTE_LLM_MODEL", "indianlegal")
        self.api_key = api_key if api_key is not None else os.getenv("REMOTE_LLM_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def ensure_loaded(self) -> None:
        """Validate configuration (no network call). Raises if no endpoint is set
        so ``build_pipeline()`` can fall back to the stub when unconfigured."""
        if not self.url:
            raise RuntimeError(
                "RemoteLLM needs an endpoint: set REMOTE_LLM_URL (an "
                "OpenAI-compatible /v1/chat/completions URL). Use LLM=stub otherwise."
            )

    def generate(self, system: str, user: str) -> str:
        self.ensure_loaded()
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "httpx is required for the remote LLM backend. "
                "Install: pip install httpx"
            ) from exc

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = httpx.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
