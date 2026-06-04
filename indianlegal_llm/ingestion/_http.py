"""Lazy HTTP GET helper for the web-based ingestors (India Code, Indian Kanoon).

Imports the HTTP client lazily (httpx, falling back to requests) so the package
imports with the standard library alone; the web ingestors require the
``ingestion`` extra. A small, polite default User-Agent and timeout are set.
"""

from __future__ import annotations

_USER_AGENT = "IndianLegal-LLM/0.1 (+https://github.com/ChaitanyaKis/IndianLegal-LLM)"


def http_get(url: str, timeout: float = 30.0) -> str:
    """GET ``url`` and return the response text. Raises on HTTP/transport error."""
    headers = {"User-Agent": _USER_AGENT}
    try:
        import httpx

        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except ImportError:
        pass
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "httpx or requests is required for web ingestion. "
            "Install the ingestion extra: pip install -e .[ingestion]"
        ) from exc
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp.text
