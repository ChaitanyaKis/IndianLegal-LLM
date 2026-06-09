"""Turn the on-disk SC tar/PDF corpus into a cleaned, processed JSONL corpus.

Build-time only (see data_pipeline.__init__). For each year it reads the
``metadata.parquet`` rows and the matching judgment PDF from ``english.tar``,
extracts text with the permissive extractor, cleans the SCR noise, derives the
SAME ``doc_id`` the retrieval ingestor uses (nc_display -> cnr -> case_id), and
writes one record per judgment to ``data/sc/processed/year=YYYY.jsonl`` (gitignored).

Both retrieval (``indianlegal_llm.ingestion.local_sc``) and the fine-tune builder
consume this processed text, so they chunk identical text (train == inference).
PDFs with no extractable text layer are QUARANTINED (counted, not emitted) — we do
not pull AGPL/OCR in to rescue them by default.
"""

from __future__ import annotations

import json
import os
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from indianlegal_llm.ingestion._util import normalize_date, safe_id

from .pdf_text import clean_scr_text, extract_pdf_text

_LICENSE = "government-work"  # Indian court judgments: freely reproducible (CA §52(1)(q))


@dataclass
class Coverage:
    year: str = ""
    rows: int = 0
    matched_pdf: int = 0
    extracted: int = 0
    quarantined_scanned: int = 0
    missing_pdf: int = 0
    no_id: int = 0
    emitted: int = 0

    def merge(self, other: "Coverage") -> None:
        for f in ("rows", "matched_pdf", "extracted", "quarantined_scanned",
                  "missing_pdf", "no_id", "emitted"):
            setattr(self, f, getattr(self, f) + getattr(other, f))


def _read_metadata(path: Path) -> list[dict]:
    """Read a single metadata.parquet (no partition inference)."""
    import pyarrow.parquet as pq

    with open(path, "rb") as fh:
        table = pq.ParquetFile(fh).read()
    columns = table.to_pydict()
    n = table.num_rows
    return [{k: columns[k][i] for k in columns} for i in range(n)]


def _clean(value) -> str:
    return str(value).strip() if value not in (None, "") and str(value) != "None" else ""


def _member_index(tar: tarfile.TarFile) -> dict[str, str]:
    """Map a normalized member stem -> member name (handle _EN suffix, S_ prefix)."""
    index: dict[str, str] = {}
    for member in tar.getmembers():
        if not member.name.endswith(".pdf"):
            continue
        stem = member.name[: -len(".pdf")]
        for key in {stem, stem[:-3] if stem.endswith("_EN") else stem}:
            index.setdefault(key, member.name)
            if key.startswith("S_"):
                index.setdefault(key[2:], member.name)
    return index


def _find_member(index: dict[str, str], path: str) -> str | None:
    for key in (path, f"{path}_EN", f"S_{path}", f"S_{path}_EN"):
        if key in index:
            return index[key]
    return None


def _doc_id(row: dict) -> str:
    ident = _clean(row.get("nc_display")) or _clean(row.get("cnr")) or _clean(row.get("case_id"))
    sid = safe_id(ident)
    return f"sc-{sid}" if sid else ""


def _record(row: dict, text: str, year: str) -> dict:
    path = _clean(row.get("path"))
    title = _clean(row.get("title"))
    if not title:
        pet, res = _clean(row.get("petitioner")), _clean(row.get("respondent"))
        title = " v. ".join(p for p in (pet, res) if p)
    return {
        "doc_id": _doc_id(row),
        "title": title,
        "court": _clean(row.get("court")) or "Supreme Court of India",
        "date": normalize_date(_clean(row.get("decision_date"))) or year,
        "url": f"s3://indian-supreme-court-judgments/{path}" if path else "",
        "license": _LICENSE,
        "language": "en",
        "nc_display": _clean(row.get("nc_display")),
        "cnr": _clean(row.get("cnr")),
        "citation": _clean(row.get("citation")),
        "year": year,
        "text": text,
    }


def extract_year(
    sc_root: Path, year: str, *, limit: int | None = None, allow_agpl: bool = False
) -> tuple[list[dict], Coverage]:
    """Extract + clean judgments for one year. Returns (records, coverage)."""
    cov = Coverage(year=year)
    meta_path = sc_root / "metadata" / f"year={year}" / "metadata.parquet"
    tar_path = sc_root / "data" / f"year={year}" / "english" / "english.tar"
    if not meta_path.exists() or not tar_path.exists():
        return [], cov

    rows = _read_metadata(meta_path)
    cov.rows = len(rows)
    records: list[dict] = []
    with tarfile.open(tar_path, "r") as tar:
        index = _member_index(tar)
        for row in rows:
            if limit is not None and cov.emitted >= limit:
                break
            if not _doc_id(row):
                cov.no_id += 1
                continue
            member = _find_member(index, _clean(row.get("path")))
            if member is None:
                cov.missing_pdf += 1
                continue
            cov.matched_pdf += 1
            raw = tar.extractfile(member).read()
            text = extract_pdf_text(raw, allow_agpl=allow_agpl)
            if text is None:
                cov.quarantined_scanned += 1
                continue
            cov.extracted += 1
            cleaned = clean_scr_text(text)
            if len(cleaned) < 400:  # post-clean too short -> skip
                cov.quarantined_scanned += 1
                continue
            records.append(_record(row, cleaned, year))
            cov.emitted += 1
    return records, cov


def write_processed(records: list[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def load_processed(processed_dir: Path) -> Iterator[dict]:
    """Yield processed records from all year=*.jsonl under ``processed_dir``."""
    for path in sorted(Path(processed_dir).glob("year=*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _parse_years(spec: str) -> list[str]:
    """Parse "2020", "2018-2026", or "2018,2020,2022" into a list of year strings."""
    years: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            years.extend(str(y) for y in range(int(lo), int(hi) + 1))
        elif part:
            years.append(part)
    return years


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from indianlegal_llm._io import enable_utf8_output

    enable_utf8_output()
    parser = argparse.ArgumentParser(prog="python -m data_pipeline.corpus")
    parser.add_argument("--sc-root", default="data/sc", help="SC corpus root (tars + metadata)")
    parser.add_argument("--years", default="2018-2026", help='e.g. "2020", "2018-2026", "2018,2020"')
    parser.add_argument("--limit-per-year", type=int, default=None, help="cap docs/year (sampling)")
    parser.add_argument("--out", default="data/sc/processed", help="output dir for year=*.jsonl")
    parser.add_argument("--allow-agpl", action="store_true", help="build-only PyMuPDF fallback")
    args = parser.parse_args([] if argv is None else argv)

    sc_root = Path(args.sc_root)
    out_dir = Path(args.out)
    total = Coverage(year="ALL")
    for year in _parse_years(args.years):
        records, cov = extract_year(
            sc_root, year, limit=args.limit_per_year, allow_agpl=args.allow_agpl
        )
        if cov.rows == 0:
            continue
        write_processed(records, out_dir / f"year={year}.jsonl")
        total.merge(cov)
        print(
            f"year={year}: rows={cov.rows} matched={cov.matched_pdf} "
            f"extracted={cov.extracted} quarantined={cov.quarantined_scanned} "
            f"missing_pdf={cov.missing_pdf} emitted={cov.emitted}"
        )
    extract_rate = (total.extracted / total.matched_pdf * 100) if total.matched_pdf else 0.0
    print(
        f"\n=== coverage (ALL) === rows={total.rows} matched_pdf={total.matched_pdf} "
        f"extracted={total.extracted} ({extract_rate:.1f}% of matched) "
        f"quarantined_scanned={total.quarantined_scanned} missing_pdf={total.missing_pdf} "
        f"emitted={total.emitted}"
    )
    print(f"wrote processed JSONL to {out_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
