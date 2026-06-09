"""Real ingestors for the AWS Open Data Indian judgment datasets.

Confirmed from the AWS Open Data registry + the dataset docs (Athena DDL), both
hosted in ``ap-south-1`` and public via ``--no-sign-request``:

  Supreme Court : s3://indian-supreme-court-judgments
                  metadata/parquet/year=YYYY/metadata.parquet   (partition: year)
  High Court    : s3://indian-high-court-judgments
                  metadata/parquet/year=YYYY/court=XXX/metadata.parquet
                                                        (partitions: year, court)

These stream the structured-metadata parquet straight from S3 with pyarrow over
an s3fs filesystem, reading in batches and stopping at ``limit`` — the corpus is
NEVER downloaded to local disk (CLAUDE.md §5). Optional deps (pyarrow, s3fs) are
imported lazily so the package still imports with the standard library alone; the
real ingestors require ``pip install -e .[ingestion]``.

Forbidden sources (SCC Online, Manupatra, ILDC) are never touched.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from ..schemas import RawDoc
from ._util import html_to_text, normalize_date, safe_id
from .base import BaseIngestor

# --- Exact, confirmed S3 locations (do not guess; see opendata registry) -------
_REGION = "ap-south-1"
_LICENSE = "government-work"

SC_BUCKET = "indian-supreme-court-judgments"
SC_METADATA_ROOT = f"{SC_BUCKET}/metadata/parquet"

HC_BUCKET = "indian-high-court-judgments"
HC_METADATA_ROOT = f"{HC_BUCKET}/metadata/parquet"


def _parquet_module():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "pyarrow is required for S3 parquet ingestion. "
            "Install the ingestion extra: pip install -e .[ingestion]"
        ) from exc
    return pq


def _s3_filesystem(anon: bool = True):
    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "s3fs is required to stream from S3. "
            "Install the ingestion extra: pip install -e .[ingestion]"
        ) from exc
    # Public dataset: anonymous access, no AWS credentials needed.
    return s3fs.S3FileSystem(anon=anon, client_kwargs={"region_name": _REGION})


def _local_filesystem():
    import fsspec

    return fsspec.filesystem("file")


def _hive_parts(path: str) -> dict[str, str]:
    """Parse hive ``key=value`` segments (year, court) out of a parquet path.

    Empty values (e.g. a malformed ``year=`` segment) are skipped so a degenerate
    path partition can never overwrite a valid same-named data column.
    """
    parts: dict[str, str] = {}
    for segment in path.replace("\\", "/").split("/"):
        if "=" in segment:
            key, value = segment.split("=", 1)
            if value:
                parts[key] = value
    return parts


class _S3ParquetIngestor(BaseIngestor):
    """Base for streaming a hive-partitioned metadata parquet from S3.

    Subclasses set ``metadata_root`` and implement ``row_to_rawdoc``. Tests may
    pass ``root`` (a local path) and ``filesystem`` (or None for the local FS) to
    exercise the read+map path without S3 or network.
    """

    metadata_root: str = ""

    def __init__(
        self,
        limit: int = 200,
        *,
        root: str | None = None,
        filesystem: object | None = None,
        anon: bool = True,
        batch_size: int = 64,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._root = root
        self._filesystem = filesystem
        self._anon = anon
        self._batch_size = batch_size

    def _resolve_source(self) -> tuple[str, object]:
        """Return (root_path, filesystem). Local override wins for tests."""
        if self._root is not None:
            fs = self._filesystem if self._filesystem is not None else _local_filesystem()
            return self._root, fs
        return self.metadata_root, _s3_filesystem(self._anon)

    def _list_parquet_files(self, fs, root: str) -> list[str]:
        """Recursively list parquet file paths under ``root``, sorted.

        This is a metadata LIST (object keys only — no data transfer), so it is
        bandwidth-safe even for the High Court prefix's ~1.4k keys; the actual
        parquet bytes are read lazily per file in ``fetch`` and stop at ``limit``.
        """
        entries = fs.find(root)
        return sorted(e for e in entries if str(e).endswith(".parquet"))

    def fetch(self) -> Iterator[RawDoc]:
        """Stream metadata parquet file-by-file, stopping at ``limit``.

        Reading one parquet file at a time (rather than a dataset-wide scan)
        avoids cross-partition schema unification and reads only as many files /
        row-groups as ``limit`` requires — the corpus is never downloaded whole
        (CLAUDE.md §5). Hive partition values (year, court) are parsed from the
        path and take precedence over any same-named data column.
        """
        pq = _parquet_module()
        root, fs = self._resolve_source()
        yielded = 0
        for path in self._list_parquet_files(fs, root):
            if yielded >= self.limit:
                return
            parts = _hive_parts(path)
            with fs.open(path, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                for batch in parquet_file.iter_batches(batch_size=self._batch_size):
                    for row in batch.to_pylist():
                        if yielded >= self.limit:
                            return
                        merged = {**row, **parts}  # path partitions win
                        doc = self.row_to_rawdoc(merged)
                        if doc is None:
                            continue
                        yield doc
                        yielded += 1

    @staticmethod
    def row_to_rawdoc(row: Mapping) -> RawDoc | None:  # pragma: no cover - abstract
        raise NotImplementedError


def _clean(row: Mapping, key: str) -> str:
    value = row.get(key)
    return str(value).strip() if value not in (None, "") else ""


def _party_title(row: Mapping) -> str:
    """Build 'Petitioner v. Respondent' from whichever parties are present."""
    parties = [p for p in (_clean(row, "petitioner"), _clean(row, "respondent")) if p]
    return " v. ".join(parties)


class AWSSupremeCourtIngestor(_S3ParquetIngestor):
    """Supreme Court of India judgments from AWS Open Data (parquet metadata).

    NOTE (Hour M-FT1): the parquet ``raw_html`` column is the eCourts page chrome
    (a language-selector), NOT the judgment body — so this ingestor's ``text`` is
    low value for retrieval. The REAL judgment text lives in the per-year PDF tars;
    the default retrieval source is now ``local-sc`` (LocalSCIngestor), which reads
    the PDF text extracted by ``data_pipeline``. This S3 ingestor remains for
    metadata/streaming; prefer ``local-sc`` for grounded answers.
    """

    source_name = "aws-sc"
    metadata_root = SC_METADATA_ROOT

    @staticmethod
    def row_to_rawdoc(row: Mapping) -> RawDoc | None:
        # Prefer the official neutral citation (nc_display, unique + human-citable),
        # then the globally-unique CNR, then the case number as a last resort.
        ident = (
            _clean(row, "nc_display")
            or _clean(row, "cnr")
            or _clean(row, "case_id")
        )
        doc_id = safe_id(ident)
        if not doc_id:
            return None  # cannot cite a doc with no usable id
        path = _clean(row, "path")
        # The dataset ships no external URL column; the S3 object location is the
        # authoritative source pointer (CLAUDE.md §3 logs a real url, not a guess).
        url = f"s3://{SC_BUCKET}/{path}" if path else f"s3://{SC_METADATA_ROOT}"
        raw_html = _clean(row, "raw_html")
        text = html_to_text(raw_html) or _clean(row, "description") or _clean(row, "title")
        title = _clean(row, "title") or _party_title(row) or ident
        available = _clean(row, "available_languages")
        return RawDoc(
            doc_id=f"sc-{doc_id}",
            title=title,
            court=_clean(row, "court") or "Supreme Court of India",
            # decision_date preferred; fall back to the year= partition.
            date=normalize_date(_clean(row, "decision_date")) or _clean(row, "year"),
            url=url,
            # The mapped text is the English judgment HTML; available_languages
            # lists ALL renditions and is kept in metadata, not the single field.
            language="en",
            license=_LICENSE,
            text=text,
            metadata={
                "dataset": SC_BUCKET,
                "nc_display": _clean(row, "nc_display"),
                "case_id": _clean(row, "case_id"),
                "cnr": _clean(row, "cnr"),
                "citation": _clean(row, "citation"),
                "judge": _clean(row, "judge"),
                "author_judge": _clean(row, "author_judge"),
                "disposal_nature": _clean(row, "disposal_nature"),
                "available_languages": available,
                "year": _clean(row, "year"),
                "s3_path": path,
                "scraped_at": _clean(row, "scraped_at"),
            },
        )


class AWSHighCourtIngestor(_S3ParquetIngestor):
    """Indian High Court judgments (25 HCs) from AWS Open Data."""

    source_name = "aws-hc"
    metadata_root = HC_METADATA_ROOT

    @staticmethod
    def row_to_rawdoc(row: Mapping) -> RawDoc | None:
        ident = _clean(row, "cnr")
        doc_id = safe_id(ident)
        if not doc_id:
            return None
        # High Court parquet carries an explicit source URL: pdf_link.
        url = _clean(row, "pdf_link") or f"s3://{HC_METADATA_ROOT}"
        title = _clean(row, "title") or ident
        # Full text lives in PDFs (not parsed here, to honour the bandwidth rule);
        # map the metadata description/title as text. PDF text extraction is a
        # documented follow-up (stream per-doc PDFs under --limit).
        text = _clean(row, "description") or title
        return RawDoc(
            doc_id=f"hc-{doc_id}",
            title=title,
            court=_clean(row, "court_name") or _clean(row, "court_code") or "High Court (India)",
            date=normalize_date(_clean(row, "decision_date")) or _clean(row, "year"),
            url=url,
            license=_LICENSE,
            text=text,
            language="en",
            metadata={
                "dataset": HC_BUCKET,
                "cnr": _clean(row, "cnr"),
                "court_code": _clean(row, "court_code"),
                "judge": _clean(row, "judge"),
                "disposal_nature": _clean(row, "disposal_nature"),
                "date_of_registration": _clean(row, "date_of_registration"),
                "year": _clean(row, "year"),
                "court": _clean(row, "court"),
                "pdf_link": _clean(row, "pdf_link"),
            },
        )
