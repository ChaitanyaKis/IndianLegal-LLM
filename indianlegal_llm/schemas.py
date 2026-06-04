"""Core data contracts shared across every layer of IndianLegal-LLM.

These dataclasses are the *contracts* that flow through the pipeline:

    RawDoc            ingestion produces these
      -> Chunk        processing splits a RawDoc into these
        -> RetrievedChunk   retrieval scores + selects these for a query
          -> Citation       the answerer turns a cited chunk into one of these
            -> Answer        the final, citation-grounded result

Keep these stable. Real implementations are added behind the Base* interfaces;
the schemas they exchange should not change in a way that breaks the skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawDoc:
    """A single raw legal document as fetched by an ingestor.

    Every field after ``text`` is provenance and is required for the ingestion
    manifest mandated by CLAUDE.md (url, court, date, license).
    """

    doc_id: str
    title: str
    court: str
    date: str  # ISO-8601 date string, e.g. "2017-08-24"
    url: str
    license: str
    text: str
    language: str = "en"  # ISO-639-ish code or label, e.g. "en", "hi"
    metadata: dict = field(default_factory=dict)

    def manifest_entry(self) -> dict:
        """Provenance record for the ingestion manifest (CLAUDE.md §3)."""
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "court": self.court,
            "date": self.date,
            "language": self.language,
            "license": self.license,
        }


@dataclass
class Chunk:
    """A retrievable slice of a RawDoc.

    ``chunk_id`` is the unit of citation. It MUST be unique across the corpus and
    must not contain square brackets (the citation marker characters).
    """

    chunk_id: str
    doc_id: str
    text: str
    title: str
    court: str
    url: str
    license: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk selected by retrieval for a specific query, with its relevance score."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """A grounded reference emitted in an Answer.

    A Citation only ever exists for a chunk that was actually retrieved for the
    query (enforced by the Answerer — see CLAUDE.md §4). ``para_start``/``para_end``
    carry the cited chunk's judgment-paragraph span (None for unnumbered text),
    enabling pinpoint citation (e.g. "Puttaswamy ¶ 297").
    """

    chunk_id: str
    doc_id: str
    title: str
    court: str
    url: str
    neutral_citation: str = ""  # e.g. "2017 INSC 1"; "" when the source has none
    para_start: int | None = None
    para_end: int | None = None

    @property
    def pinpoint(self) -> str:
        """Render the paragraph pinpoint: '¶ N', '¶ N-M', or '' when unnumbered."""
        if self.para_start is None:
            return ""
        if self.para_end is None or self.para_end == self.para_start:
            return f"¶ {self.para_start}"
        return f"¶ {self.para_start}-{self.para_end}"

    @property
    def reference(self) -> str:
        """Full human-readable citation: '<title>, <neutral citation> <pinpoint>'.

        Title for readability, neutral citation for verifiability, pinpoint to the
        paragraph. Each component is omitted gracefully when absent. Built ONLY
        from the retrieved chunk's trusted metadata — never from model free text.
        """
        label = self.title
        if self.neutral_citation:
            label = f"{label}, {self.neutral_citation}"
        if self.pinpoint:
            label = f"{label} {self.pinpoint}"
        return label


@dataclass
class Answer:
    """The final result returned to a caller.

    Invariant (trust property): ``refused`` is True iff ``citations`` is empty.
    A non-refused answer always carries at least one valid citation.
    """

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False

    @property
    def is_grounded(self) -> bool:
        """True when the answer makes a legal claim backed by >=1 valid citation."""
        return (not self.refused) and bool(self.citations)
