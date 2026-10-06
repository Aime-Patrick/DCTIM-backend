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
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class GroundingValidationError(ValueError):
    """Raised when a generated answer cannot be tied to retrieved evidence."""


def validate_analysis_references(
    analysis: object,
    evidence_chunk_ids: set[str],
) -> None:
    """Reject populated analysis sections that have no valid source references."""
    if not isinstance(analysis, dict):
        raise GroundingValidationError("analysis response was not an object")

    for field in ("risks", "recommendations", "metrics"):
        items = analysis.get(field, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                raise GroundingValidationError(f"analysis field '{field}' contains an invalid item")
            refs = item.get("evidence_refs")
            if not isinstance(refs, list) or not refs:
                raise GroundingValidationError(
                    f"analysis field '{field}' contains an uncited item"
                )
            invalid = [str(ref) for ref in refs if str(ref) not in evidence_chunk_ids]
            if invalid:
                raise GroundingValidationError(
                    f"analysis field '{field}' cites unavailable chunks: {', '.join(invalid)}"
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
    # Preserve line boundaries because they carry meaning in Markdown tables,
    # extracted pages and DOCX structure.  Collapse only repeated horizontal
    # whitespace and excessive blank lines.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


def wrap_evidence_block(index: int, title: str, source_type: str, score: float, text: str) -> str:
    safe = sanitize_untrusted_text(text, max_chars=12000)
    return (
        f"[{index}] UNTRUSTED_SOURCE ({source_type}: {title}, score={score:.4f})\n"
        f"<<<EVIDENCE\n{safe}\nEVIDENCE>>>"
    )


def validate_grounded_answer(answer: str, evidence_count: int) -> str:
    """Require the generated answer to cite retrieved evidence.

    Validates that:
    - The answer is not empty.
    - At least one citation exists.
    - All cited block numbers are in range.

    Per-sentence citation enforcement was removed: it rejected otherwise correct
    answers where a model writes a flowing paragraph and places the citation at
    the end of the paragraph rather than each individual sentence. The key
    grounding guarantee — that the answer cites workspace evidence and does not
    cite invented blocks — is preserved.
    """
    if not answer or not answer.strip():
        raise GroundingValidationError("the answer was empty")
    if is_insufficient_answer(answer):
        return answer.strip()

    references = {int(value) for value in _CITATION_PATTERN.findall(answer)}
    if not references:
        raise GroundingValidationError("the answer contained no evidence citations")

    invalid = sorted(reference for reference in references if reference < 1 or reference > evidence_count)
    if invalid:
        raise GroundingValidationError(
            f"the answer cited unavailable evidence blocks: {', '.join(map(str, invalid))}"
        )
    return answer.strip()
