"""Application surfaces: CLI (zero-dep), API (FastAPI), demo (Gradio).

All three obtain their pipeline from `build_pipeline()` and never wire components
directly. The API and demo guard their optional imports so the package always
imports cleanly with only the standard library.
"""
