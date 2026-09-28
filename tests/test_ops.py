"""Phase 5 ops/security unit tests."""
from __future__ import annotations

from app.core.rate_limit import SlidingWindowRateLimiter
from app.modules.rag.upload_validation import (
    UploadValidationError,
    validate_extracted_text,
    validate_upload_bytes,
)


def test_rate_limiter_blocks_after_budget() -> None:
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60)
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_validate_upload_rejects_bad_extension() -> None:
    try:
        validate_upload_bytes("malware.exe", b"MZ", max_bytes=1024)
        raise AssertionError("expected UploadValidationError")
    except UploadValidationError as exc:
        assert "Unsupported" in str(exc)


def test_validate_pdf_magic_bytes() -> None:
    try:
        validate_upload_bytes("report.pdf", b"not-a-pdf", max_bytes=1024)
        raise AssertionError("expected UploadValidationError")
    except UploadValidationError as exc:
        assert "does not match" in str(exc)

    name = validate_upload_bytes("report.pdf", b"%PDF-1.4 content", max_bytes=1024)
    assert name == "report.pdf"


def test_validate_extracted_text_bounds() -> None:
    try:
        validate_extracted_text("x" * 100, max_chars=50)
        raise AssertionError("expected UploadValidationError")
    except UploadValidationError:
        pass
    assert validate_extracted_text("  hello  ", max_chars=50) == "hello"
