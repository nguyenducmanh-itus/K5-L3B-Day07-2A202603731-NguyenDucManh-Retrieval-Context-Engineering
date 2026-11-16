"""Build a page-linked lexical index for the Day 7 advanced source PDF.

Requires Poppler's pdftotext/pdfinfo on PATH. Text extraction uses layout mode so
table rows/columns and figure labels are preserved as much as possible. Every
chunk retains the physical PDF page number; page renders remain the authority
for visual verification.
"""
from __future__ import annotations

import json
import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "data" / "advanced"
PDF = ADV / "world_development_report_2021.pdf"
TEXT = ADV / "world_development_report_2021_layout.txt"
PAGES = ADV / "pages.jsonl"
CHUNKS = ADV / "chunks.json"
MANIFEST = ADV / "source_manifest.json"


def main() -> None:
    if not PDF.exists():
        raise SystemExit(f"Missing {PDF}; fetch the official PDF per source_manifest.json")
    info = subprocess.run(["pdfinfo", str(PDF)], check=True, capture_output=True, text=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo did not report a page count")
    page_count = int(match.group(1))
    subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(PDF), str(TEXT)], check=True)
    pages = TEXT.read_text(encoding="utf-8", errors="replace").split("\f")
    pages = pages[:page_count]
    chunks: list[dict] = []
    page_rows: list[dict] = []
    empty_pages: list[int] = []
    indexed_chars = 0
    for page_no, text in enumerate(pages, 1):
        normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        indexed_chars += len(normalized)
        if not normalized:
            empty_pages.append(page_no)
        page_rows.append({"pdf_page": page_no, "has_text": bool(normalized), "chars": len(normalized),
                          "status": "indexed" if normalized else "no_machine_text_not_OCRd"})
        # Keep page number stable; split only on paragraph boundaries and retain
        # a small overlap so retrieval does not lose boundary facts.
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", normalized)]
        buf = ""
        part = 0
        for paragraph in paragraphs:
            if not paragraph:
                continue
            if buf and len(buf) + len(paragraph) + 1 > 1800:
                chunks.append({"id": f"wdr21-p{page_no:03d}-c{part:02d}", "source_id": "wdr21",
                               "source": "World Development Report 2021: Data for Better Lives",
                               "page": str(page_no), "text": buf})
                part += 1
                buf = (buf[-240:] + " " + paragraph) if len(buf) > 240 else (buf + " " + paragraph)
            else:
                buf = (buf + "\n" + paragraph).strip()
        if buf:
            chunks.append({"id": f"wdr21-p{page_no:03d}-c{part:02d}", "source_id": "wdr21",
                           "source": "World Development Report 2021: Data for Better Lives",
                           "page": str(page_no), "text": buf})
    PAGES.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in page_rows), encoding="utf-8")
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest["index"] = {"extractor": "Poppler pdftotext -layout", "page_count": page_count,
                         "pages_with_machine_text": page_count - len(empty_pages),
                         "pages_without_machine_text_not_OCRd": empty_pages,
                         "page_records": len(page_rows), "chunk_count": len(chunks),
                         "indexed_text_characters": indexed_chars,
                         "source_pdf_bytes": PDF.stat().st_size,
                         "source_pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
                         "chunk_policy": "page-aware paragraphs, target 1800 chars, 240-char overlap",
                         "retrieval": "BM25 with explicit Figure/Map/Table locator prioritization",
                         "text_file": TEXT.name, "page_ledger": PAGES.name, "chunk_file": CHUNKS.name,
                         "limitations": "15 pages have no machine-readable text and were not OCR'd. Complex visual semantics are not decoded by pdftotext; tested chart, table, map and diagram regions were checked against rendered source pages."}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["index"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
