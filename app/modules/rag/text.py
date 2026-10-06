"""Structure- and table-aware text chunking for RAG retrieval.

Features:
- Preserves markdown headers and structural hierarchy.
- Preserves table / CSV header rows when tabular data spans multiple chunks.
- Splits at natural boundaries: headings -> paragraphs -> sentences -> words.
- Maintains configurable overlap without truncating words mid-syllable.
"""
from __future__ import annotations

import re

# Boundary patterns in descending order of structural priority.
_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6}\s+.+)$")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.+\|\s*$|^\s*[^,\t]+(?:[,\t][^,\t]+)+\s*$")
_SOURCE_MARKER_PATTERN = re.compile(
    r"\[(?:Page\s+\d+|DOCX\s+(?:paragraph|table|header|footer)(?:\s+section)?\s+\d+)\]",
    re.IGNORECASE,
)


def source_locations(text: str) -> list[str]:
    """Return stable source markers carried into a chunk for citation metadata."""
    seen: set[str] = set()
    locations: list[str] = []
    for marker in _SOURCE_MARKER_PATTERN.findall(text):
        if marker not in seen:
            seen.add(marker)
            locations.append(marker)
    return locations


def _is_table_row(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if line.startswith("|") and line.endswith("|"):
        return True
    if "\t" in line or ("," in line and not line.endswith(".")):
        return True
    return False


def _detect_table_header(lines: list[str]) -> tuple[str | None, list[str]]:
    """Check if the start of *lines* forms a table and extract its header."""
    if not lines or not _is_table_row(lines[0]):
        return None, lines

    first_line = lines[0].strip()
    if len(lines) > 1 and (lines[1].strip().startswith("|-") or lines[1].strip().startswith("|:-") or lines[1].strip().startswith("| -")):
        # Markdown table with separator line
        header = f"{lines[0].strip()}\n{lines[1].strip()}"
        body = lines[2:]
        return header, body
    elif len(lines) > 1 and ("\t" in first_line or "," in first_line):
        # CSV / TSV with header row
        return first_line, lines[1:]

    return None, lines


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 120) -> list[str]:
    """Split text into bounded, structure-preserving overlapping chunks.

    Parameters
    ----------
    text:
        Input document text.
    max_chars:
        Maximum character length per chunk (must be positive).
    overlap:
        Overlap character budget between consecutive chunks (0 <= overlap < max_chars).

    Returns
    -------
    list[str]
        List of cleaned chunk strings.
    """
    if not text or not text.strip():
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")

    # Normalize only line endings.  Do not collapse blank lines or strip every
    # line: those boundaries can represent document structure and table rows.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []

    # First, split by double newlines or markdown headings into semantic sections
    raw_sections: list[str] = []
    # Split on markdown headings or double newlines
    tokens = re.split(r"(\n\s*#{1,6}\s+[^\n]+|\n\n+)", normalized)
    current_acc = ""
    for token in tokens:
        if not token:
            continue
        if token.startswith("\n") and ("#" in token or "\n\n" in token):
            if current_acc.strip():
                raw_sections.append(current_acc.strip())
            current_acc = token.lstrip()
        else:
            current_acc += token
    if current_acc.strip():
        raw_sections.append(current_acc.strip())

    chunks: list[str] = []
    current_chunk = ""
    current_header: str | None = None
    table_header: str | None = None

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        # Check if section starts with a markdown header
        if section.startswith("#"):
            first_line, _, rest = section.partition("\n")
            current_header = first_line.strip()
            section_content = rest.strip()
        else:
            section_content = section

        # Detect table headers inside section
        section_lines = [line.strip() for line in section_content.splitlines() if line.strip()]
        detected_th, remaining_lines = _detect_table_header(section_lines)
        if detected_th:
            table_header = detected_th

        # If the entire section fits in current chunk:
        candidate = f"{current_chunk}\n\n{section}".strip() if current_chunk else section
        if len(candidate) <= max_chars:
            current_chunk = candidate
            continue

        # If current chunk has accumulated content, push it
        if current_chunk:
            chunks.append(current_chunk)
            # Retain overlap from end of current chunk
            overlap_tail = current_chunk[-overlap:].strip() if overlap > 0 else ""
            # If we had a table header or heading context, we can re-anchor
            current_chunk = ""
            if table_header and any(_is_table_row(line) for line in section_lines):
                current_chunk = f"{table_header}\n"

        # If this individual section itself exceeds max_chars, split by lines / sentences / words
        if len(section) > max_chars:
            lines = section.splitlines()
            sub_chunk = ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                line_candidate = f"{sub_chunk}\n{line}".strip() if sub_chunk else line
                # Prepend context header if beginning a fresh chunk
                if not sub_chunk and current_header and not line.startswith("#"):
                    prefix = f"[{current_header}]\n"
                    if len(prefix) + len(line_candidate) <= max_chars:
                        line_candidate = f"{prefix}{line_candidate}"

                if len(line_candidate) <= max_chars:
                    sub_chunk = line_candidate
                else:
                    if sub_chunk:
                        chunks.append(sub_chunk)
                        sub_chunk = ""
                    # If a single line exceeds max_chars, split by sentences or sliding window
                    if len(line) > max_chars:
                        sentences = _SENTENCE_PATTERN.split(line)
                        s_chunk = ""
                        for s in sentences:
                            s_cand = f"{s_chunk} {s}".strip() if s_chunk else s
                            if len(s_cand) <= max_chars:
                                s_chunk = s_cand
                            else:
                                if s_chunk:
                                    chunks.append(s_chunk)
                                    s_chunk = ""
                                if len(s) > max_chars:
                                    # Fallback character window
                                    start = 0
                                    while start < len(s):
                                        end = min(start + max_chars, len(s))
                                        if end < len(s):
                                            space_b = s.rfind(" ", start + (max_chars // 2), end)
                                            if space_b > start:
                                                end = space_b
                                        chunks.append(s[start:end].strip())
                                        if end >= len(s):
                                            break
                                        start = max(end - overlap, start + 1)
                                else:
                                    s_chunk = s
                        if s_chunk:
                            sub_chunk = s_chunk
                    else:
                        sub_chunk = line
            if sub_chunk:
                current_chunk = sub_chunk
        else:
            current_chunk = section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Final cleanup: drop empty chunks and ensure no trailing whitespace.
    # The original implementation calculated overlap_tail but discarded it.
    # Add a bounded, word-safe tail to each following chunk when there is room.
    raw_chunks = [c.strip() for c in chunks if c.strip()]
    if overlap <= 0 or len(raw_chunks) < 2:
        return raw_chunks

    result = [raw_chunks[0]]
    for previous, current in zip(raw_chunks, raw_chunks[1:]):
        tail = previous[-overlap:].strip()
        if tail and len(previous) > overlap:
            first_space = tail.find(" ")
            if first_space >= 0:
                tail = tail[first_space + 1 :].strip()
        available = max_chars - len(current) - 1
        if tail and available > 0:
            tail = tail[-available:].lstrip()
            if tail:
                current = f"{tail}\n{current}"
        result.append(current)
    return result
