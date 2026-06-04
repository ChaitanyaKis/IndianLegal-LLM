"""Stub ingestor: two real Supreme Court of India judgments, baked in as text.

These are landmark, public-record judgments. The text below is an original,
factual summary written for the skeleton — not a copy of any proprietary headnote
or commentary. Provenance points at Indian Kanoon (a government-works source that
is license-clean per CLAUDE.md §3).

When this stub is replaced, the real ingestor streams from AWS Open Data / India
Code / Indian Kanoon behind the same :class:`BaseIngestor` interface.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..schemas import RawDoc
from .base import BaseIngestor

_GOV_LICENSE = "Public record (Indian Kanoon - government works)"
_SUPREME_COURT = "Supreme Court of India"


_PUTTASWAMY_TEXT = """\
Justice K. S. Puttaswamy (Retd.) and another versus Union of India and others.

1. In this decision a nine-judge bench of the Supreme Court of India held that the
right to privacy is a fundamental right protected as an intrinsic part of the
right to life and personal liberty under Article 21 of the Constitution, and as
a part of the freedoms guaranteed by Part III of the Constitution.

2. The Court overruled the earlier views in M. P. Sharma and Kharak Singh to the
extent that those decisions held that privacy is not a fundamental right. The
bench unanimously affirmed that privacy is a constitutionally protected right
that emerges primarily from the guarantee of life and personal liberty.

3. Privacy includes the preservation of personal intimacies, the sanctity of family
life, the protection of individual autonomy, dignity, and the right of every
person to make decisions about matters central to their own life. Informational
privacy is a facet of this fundamental right, and the State must put in place a
robust data protection regime to safeguard it.

4. The right to privacy is not absolute. Any restriction by the State must satisfy
the tests of legality, a legitimate State aim, and proportionality. The judgment
establishes that dignity and liberty are the foundation on which the fundamental
right to privacy rests in India.
"""

_KESAVANANDA_TEXT = """\
Kesavananda Bharati Sripadagalvaru and others versus State of Kerala and another.

1. In this decision a thirteen-judge bench of the Supreme Court of India laid down
the basic structure doctrine. The Court held that Parliament has wide power to
amend the Constitution under Article 368, but that this power does not extend to
altering, damaging, or destroying the basic structure or essential framework of
the Constitution.

2. The basic structure doctrine means that while individual provisions of the
Constitution may be amended, the fundamental architecture of the Constitution
must be preserved. Features identified as part of the basic structure include
the supremacy of the Constitution, the rule of law, the separation of powers,
judicial review, the federal character of the polity, and the protection of
fundamental rights.

3. The judgment partly overruled Golak Nath and upheld the validity of the
constitutional amendments at issue, while subjecting the amending power itself to
the implied limitation that the essential identity of the Constitution cannot be
abrogated. This doctrine remains the cornerstone of Indian constitutional law and
constrains the amending power of Parliament to this day.
"""


class StubIngestor(BaseIngestor):
    """Yields two landmark SC judgments. Pure stdlib, zero network access."""

    source_name = "stub:indian-kanoon"

    def fetch(self) -> Iterator[RawDoc]:
        yield RawDoc(
            doc_id="puttaswamy-2017",
            title="K. S. Puttaswamy v. Union of India",
            court=_SUPREME_COURT,
            date="2017-08-24",
            url="https://indiankanoon.org/doc/91938676/",
            license=_GOV_LICENSE,
            text=_PUTTASWAMY_TEXT,
            language="en",
            metadata={"citation": "(2017) 10 SCC 1", "bench": 9, "topic": "privacy"},
        )
        yield RawDoc(
            doc_id="kesavananda-1973",
            title="Kesavananda Bharati v. State of Kerala",
            court=_SUPREME_COURT,
            date="1973-04-24",
            url="https://indiankanoon.org/doc/257876/",
            license=_GOV_LICENSE,
            text=_KESAVANANDA_TEXT,
            language="en",
            metadata={
                "citation": "(1973) 4 SCC 225",
                "bench": 13,
                "topic": "basic-structure",
            },
        )
