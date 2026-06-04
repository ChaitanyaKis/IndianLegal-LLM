"""Exception classification shared by the ingestion fallback logic.

These are programmer-error types that must always surface (propagate) rather than
be masked as "ingestor unavailable" and silently fall back to the stub — even in
non-strict mode. Degradable conditions (missing [ingestion] extra, network,
credentials, malformed data) are everything else.
"""

PROGRAMMER_ERRORS = (AttributeError, TypeError, NameError, KeyError, IndexError)
