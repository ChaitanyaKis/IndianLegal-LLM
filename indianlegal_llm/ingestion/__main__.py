"""Enables `python -m indianlegal_llm.ingestion ...` to run the ingestion CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
