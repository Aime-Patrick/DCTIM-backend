"""Upload validation helpers (extension, size, basic content safety)."""
from __future__ import annotations

from pathlib import Path

from .infrastructure.extractors import SUPPORTED_EXTENSIONS

# Minimal magic-byte signatures for common uploads (best-effort; not antivirus).
_MAGIC_PREFIXES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",),  # zip container
    ".json": (),  # text
    ".jsonl": (),
    ".txt": (),
    ".md": (),
    ".csv": (),
}


class UploadValidationError(ValueError):
    """Raised when an upload fails validation before ingestion."""


def validate_upload_filename(filename: str) -> str:
    name = Path(filename or "upload").name
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise UploadValidationError("Invalid upload filename.")
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return name


def validate_upload_bytes(filename: str, data: bytes, *, max_bytes: int) -> str:
    """Validate size and basic content; return normalized filename."""
    name = validate_upload_filename(filename)
    if not data:
        raise UploadValidationError("Uploaded file is empty.")
    if len(data) > max_bytes:
        mb = max_bytes // 1024 // 1024
        raise UploadValidationError(
            f"File exceeds the {mb} MB limit ({len(data) // 1024 // 1024} MB received)."
        )
    if b"\x00" in data[:4096] and not name.lower().endswith((".pdf", ".docx")):
        # Null bytes in early text payloads are usually not legitimate text docs.
        raise UploadValidationError("File contains binary null bytes unexpected for this type.")

    ext = "." + name.rsplit(".", 1)[-1].lower()
    prefixes = _MAGIC_PREFIXES.get(ext, ())
    if prefixes and not any(data.startswith(prefix) for prefix in prefixes):
        raise UploadValidationError(
            f"File content does not match expected format for '{ext}'."
        )
    return name


def validate_extracted_text(content: str, *, max_chars: int) -> str:
    text = content.replace("\x00", " ").strip()
    if not text:
        raise UploadValidationError("No extractable text found in the uploaded file.")
    if len(text) > max_chars:
        raise UploadValidationError(
            f"Extracted text exceeds the {max_chars} character limit."
        )
    return text
