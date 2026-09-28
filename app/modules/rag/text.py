from __future__ import annotations


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into bounded, slightly overlapping chunks for retrieval."""

    normalized = " ".join(text.split())
    if not normalized:
        return []
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + (max_chars // 2), end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks

