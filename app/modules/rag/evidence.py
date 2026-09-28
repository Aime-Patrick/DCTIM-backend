"""Shared insufficient-evidence helpers and untrusted-document sanitization."""
from __future__ import annotations

import re
from collections.abc import Sequence

from .domain import RetrievedChunk

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find relevant evidence in the indexed workspace."
)

_INSUFFICIENT_MARKERS = (
    "could not find relevant evidence",
    "insufficient evidence",
    "do not support an answer",
    "no relevant evidence",
    "evidence is insufficient",
)

# Common prompt-injection / instruction-override phrases in untrusted docs.
_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"),
    re.compile(r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+instructions?"),
    re.compile(r"(?i)system\s*prompt\s*:"),
    re.compile(r"(?i)you\s+are\s+now\s+(?:a|an|the)\b"),
    re.compile(r"(?i)new\s+instructions?\s*:"),
    re.compile(r"(?i)</?\s*system\s*>"),
)


def is_insufficient_answer(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _INSUFFICIENT_MARKERS)


def filter_by_min_score(
    contexts: Sequence[RetrievedChunk],
    min_score: float,
) -> list[RetrievedChunk]:
    """Drop weak retrieval hits so generation is not fed noise."""
    if min_score <= 0:
        return list(contexts)
    return [match for match in contexts if match.score >= min_score]


def sanitize_untrusted_text(text: str, *, max_chars: int = 4000) -> str:
    """Neutralize prompt-injection cues and bound evidence size.

    Retrieved document text is untrusted. We do not execute it as instructions;
    we strip common override phrases and wrap the remainder for the LLM prompt.
    """
    cleaned = text.replace("\x00", " ")
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered]", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


def wrap_evidence_block(index: int, title: str, source_type: str, score: float, text: str) -> str:
    safe = sanitize_untrusted_text(text)
    return (
        f"[{index}] UNTRUSTED_SOURCE ({source_type}: {title}, score={score:.4f})\n"
        f"<<<EVIDENCE\n{safe}\nEVIDENCE>>>"
    )
