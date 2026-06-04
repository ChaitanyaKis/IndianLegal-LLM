"""Processing layer: turn a RawDoc into retrievable Chunks.

Interface: :class:`~indianlegal_llm.processing.base.BaseProcessor`.
Skeleton stub: :class:`~indianlegal_llm.processing.stub.StubProcessor`.
"""

from .base import BaseProcessor
from .stub import StubProcessor

__all__ = ["BaseProcessor", "StubProcessor"]
