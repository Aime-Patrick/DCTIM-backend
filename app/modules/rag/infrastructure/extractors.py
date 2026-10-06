"""Text extractors for uploaded document files.

Each extractor receives raw ``bytes`` and returns a plain-text ``str``.
The dispatcher ``extract_text`` routes by file extension.

Supported formats
-----------------
- ``.txt`` / ``.md``   — decoded as UTF-8 (with latin-1 fallback)
- ``.json`` / ``.jsonl`` — flattened to key: value lines
- ``.csv``             — rows joined as tab-separated lines
- ``.pdf``             — text layer via pypdf (no OCR)
- ``.docx``            — paragraph text via python-docx

Adding a new format
-------------------
Register a function with the signature ``(data: bytes) -> str`` in
``_EXTRACTORS`` below.  Raise ``ExtractionError`` for unrecoverable failures.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from typing import Callable


EXTRACTION_VERSION = "structured-v2"


@dataclass(frozen=True)
class ExtractedDocument:
    """Canonical text plus the provenance manifest produced by an extractor.

    The original upload is stored separately.  This representation is the
    loss-aware text used for chunking and retrieval, and its manifest is kept
    with the document metadata so extraction can be audited and re-run.
    """

    text: str
    metadata: dict[str, object]


class ExtractionError(ValueError):
    """Raised when a file cannot be parsed into usable text."""


# ---------------------------------------------------------------------------
# Individual extractors
# ---------------------------------------------------------------------------

def _extract_plain(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _extract_json(data: bytes) -> str:
    text = _extract_plain(data)
    lines: list[str] = []

    def _flatten(obj: object, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}.")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}{i}.")
        else:
            lines.append(f"{prefix.rstrip('.')}: {obj}")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Invalid JSON: {exc}") from exc

    if isinstance(parsed, list):
        # JSONL-style list: flatten each item separately
        for item in parsed:
            _flatten(item)
            lines.append("")  # blank line between records
    else:
        _flatten(parsed)

    return "\n".join(lines).strip()


def _extract_jsonl(data: bytes) -> str:
    text = _extract_plain(data)
    parts: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"Invalid JSON on line {i}: {exc}") from exc
        # Concatenate all string values from each JSON object
        values = [str(v) for v in (obj.values() if isinstance(obj, dict) else [obj])]
        parts.append(" ".join(values))
    return "\n".join(parts)


def _extract_csv(data: bytes) -> str:
    text = _extract_plain(data)
    reader = csv.reader(io.StringIO(text))
    rows: list[str] = []
    for row in reader:
        rows.append("\t".join(cell.strip() for cell in row))
    return "\n".join(rows)


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError("pypdf is not installed; cannot extract PDF text.") from exc

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    if not pages:
        raise ExtractionError(
            "No extractable text layer found in this PDF. "
            "Scanned PDFs require OCR which is not yet supported."
        )
    return "\n\n".join(pages)


def _extract_pdf_document(data: bytes) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError("pypdf is not installed; cannot extract PDF text.") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF: {exc}") from exc

    page_blocks: list[str] = []
    pages_with_text: list[int] = []
    pages_needing_ocr: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = (page.extract_text() or "").replace("\x00", "").strip()
        except Exception as exc:
            raise ExtractionError(f"Could not extract text from PDF page {page_number}: {exc}") from exc

        if page_text:
            pages_with_text.append(page_number)
            page_blocks.append(f"[Page {page_number}]\n{page_text}")
        else:
            pages_needing_ocr.append(page_number)
            page_blocks.append(
                f"[Page {page_number}]\n[No text layer extracted; OCR is required for this page.]"
            )

    if not pages_with_text:
        raise ExtractionError(
            "No extractable text layer found in this PDF. "
            "Scanned PDFs require OCR which is not yet supported."
        )

    return ExtractedDocument(
        text="\n\n".join(page_blocks),
        metadata={
            "format": "pdf",
            "extraction_method": "pypdf-text-layer",
            "extraction_version": EXTRACTION_VERSION,
            "page_count": len(reader.pages),
            "pages_with_text": pages_with_text,
            "pages_needing_ocr": pages_needing_ocr,
            "ocr_required": bool(pages_needing_ocr),
        },
    )


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise ExtractionError("python-docx is not installed; cannot extract DOCX text.") from exc

    doc = DocxDocument(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ExtractionError("No text paragraphs found in this DOCX file.")
    return "\n\n".join(paragraphs)


def _paragraph_markdown(paragraph: object, index: int) -> str:
    """Render a DOCX paragraph without discarding heading/list semantics."""
    text = str(getattr(paragraph, "text", "")).replace("\x00", "").strip()
    if not text:
        return ""

    style_name = str(getattr(getattr(paragraph, "style", None), "name", "")).lower()
    if style_name.startswith("heading"):
        match = re.search(r"(\d+)", style_name)
        level = min(6, max(1, int(match.group(1)) if match else 1))
        return f"{'#' * level} {text}"
    if "list bullet" in style_name:
        return f"- {text}"
    if "list number" in style_name:
        return f"1. {text}"
    return text


def _table_markdown(table: object) -> str:
    rows: list[list[str]] = []
    for row in getattr(table, "rows", []):
        cells = []
        for cell in getattr(row, "cells", []):
            cell_text = " / ".join(
                line.strip()
                for line in str(getattr(cell, "text", "")).splitlines()
                if line.strip()
            )
            cells.append(cell_text.replace("|", "\\|"))
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body])


def _extract_docx_document(data: bytes) -> ExtractedDocument:
    try:
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise ExtractionError("python-docx is not installed; cannot extract DOCX text.") from exc

    try:
        doc = DocxDocument(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not open DOCX: {exc}") from exc

    blocks: list[str] = []
    paragraph_count = 0
    table_count = 0

    # Iterate the document body in source order.  Reading doc.paragraphs and
    # doc.tables separately loses the relationship between paragraphs/tables.
    for element in doc.element.body.iterchildren():
        if element.tag.endswith("}p"):
            paragraph_count += 1
            rendered = _paragraph_markdown(Paragraph(element, doc), paragraph_count)
            if rendered:
                blocks.append(f"[DOCX paragraph {paragraph_count}]\n{rendered}")
        elif element.tag.endswith("}tbl"):
            table_count += 1
            rendered = _table_markdown(Table(element, doc))
            if rendered:
                blocks.append(f"[DOCX table {table_count}]\n{rendered}")

    # Headers and footers are outside the document body and would otherwise be
    # silently lost.  Include them with explicit provenance markers.
    header_count = 0
    footer_count = 0
    for section_number, section in enumerate(doc.sections, start=1):
        for label, container in (("header", section.header), ("footer", section.footer)):
            paragraphs = [p for p in container.paragraphs if p.text.strip()]
            if not paragraphs:
                continue
            if label == "header":
                header_count += 1
            else:
                footer_count += 1
            rendered = "\n".join(
                item for index, paragraph in enumerate(paragraphs, start=1)
                if (item := _paragraph_markdown(paragraph, index))
            )
            blocks.append(f"[DOCX {label} section {section_number}]\n{rendered}")

    text = "\n\n".join(blocks).strip()
    if not text:
        raise ExtractionError("No text, tables, headers, or footers found in this DOCX file.")

    return ExtractedDocument(
        text=text,
        metadata={
            "format": "docx",
            "extraction_method": "python-docx-structured",
            "extraction_version": EXTRACTION_VERSION,
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "header_count": header_count,
            "footer_count": footer_count,
        },
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, Callable[[bytes], str]] = {
    ".txt":   _extract_plain,
    ".md":    _extract_plain,
    ".json":  _extract_json,
    ".jsonl": _extract_jsonl,
    ".csv":   _extract_csv,
    ".pdf":   _extract_pdf,
    ".docx":  _extract_docx,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXTRACTORS)


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from *data* based on *filename* extension.

    Parameters
    ----------
    filename:
        Original filename including extension (used only for routing).
    data:
        Raw file bytes.

    Returns
    -------
    str
        Extracted plain text, normalised to UTF-8.

    Raises
    ------
    ExtractionError
        If the file type is unsupported or the file cannot be parsed.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ExtractionError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext == ".pdf":
        extracted = _extract_pdf_document(data).text
    elif ext == ".docx":
        extracted = _extract_docx_document(data).text
    else:
        extracted = extractor(data)
    if not extracted.strip():
        raise ExtractionError(f"No text could be extracted from '{filename}'.")
    return extracted


def extract_document(filename: str, data: bytes) -> ExtractedDocument:
    """Extract canonical text and a provenance manifest from an uploaded file."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        return _extract_pdf_document(data)
    if ext == ".docx":
        return _extract_docx_document(data)

    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ExtractionError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    text = extractor(data)
    if not text.strip():
        raise ExtractionError(f"No text could be extracted from '{filename}'.")
    return ExtractedDocument(
        text=text,
        metadata={
            "format": ext.removeprefix("."),
            "extraction_method": "text-normalized",
            "extraction_version": EXTRACTION_VERSION,
        },
    )
