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
from typing import Callable


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
    extracted = extractor(data)
    if not extracted.strip():
        raise ExtractionError(f"No text could be extracted from '{filename}'.")
    return extracted
