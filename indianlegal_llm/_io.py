"""Small stdlib helper to keep console output robust across platforms.

Real Indian legal data (judgment text, party/judge names) will contain em-dashes,
smart quotes, and Indian-language/accented characters. On a Windows OEM codepage
(cp437/cp850) or a redirected pipe under ascii, printing those would raise
UnicodeEncodeError and abort the CLI or the green-build harness. This switches the
output streams to UTF-8 (best effort) so that never happens.
"""

from __future__ import annotations

import sys


def enable_utf8_output() -> None:
    """Best-effort: reconfigure stdout/stderr to UTF-8 with a safe error handler.

    No-op on streams that don't support ``reconfigure`` (e.g. already-wrapped
    streams in some test harnesses).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass
