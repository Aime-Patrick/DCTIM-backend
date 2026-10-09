from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..indicators.domain import CaseIndicator
from ..rag.application import RagService


@dataclass(frozen=True)
class BaselineProposal:
    indicator_id: str
    indicator_name: str
    baseline_value: float
    current_value: float | None
    uncertainty: float | None
    quality_status: str
    source_refs: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class BaselineProposalPackage:
    proposal_id: str
    case_id: str
    prompt: str
    generation_mode: str
    summary: str
    updates: tuple[BaselineProposal, ...]
    warnings: tuple[str, ...]


def build_baseline_proposal(
    workspace_id: str,
    case_id: str,
    indicators: list[CaseIndicator],
    source_ids: tuple[str, ...],
    prompt: str,
    rag: RagService | None = None,
) -> BaselineProposalPackage:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("prompt is required")

    analysis: dict[str, Any] = {}
    if rag is not None and source_ids and indicators:
        indicator_context = "\n".join(
            f"- {item.id}: {item.name} ({item.unit}) — {item.definition}"
            for item in indicators
        )
        query = (
            "Extract only explicit baseline or current numeric values for the listed settlement "
            "indicators from the linked evidence. Never estimate or infer a value. "
            "Return each metric label exactly as one of the listed indicator names and cite the "
            "supporting chunk IDs in evidence_refs.\n"
            f"Indicators:\n{indicator_context}\n"
            f"User request: {normalized_prompt}"
        )
        try:
            analysis = rag.analyze(
                workspace_id,
                query,
                top_k=8,
                document_ids=source_ids,
            ).analysis
        except Exception:
            analysis = {}

    updates = tuple(_match_metrics(indicators, analysis.get("metrics")))
    warnings: list[str] = []
    if not source_ids:
        warnings.append("Link an evidence source before asking DC-TIM to extract a baseline.")
    if not indicators:
        warnings.append("Approve at least one indicator before requesting baseline extraction.")
    if not updates:
        warnings.append(
            "No explicit numeric baseline was found. Upload a measured dataset or provide a source with stated values."
        )
    if updates and any(not item.source_refs for item in updates):
        warnings.append("Some proposed values still need exact evidence references before approval.")

    mode = "model_assisted" if updates else "evidence_not_found"
    return BaselineProposalPackage(
        proposal_id=uuid4().hex,
        case_id=case_id,
        prompt=normalized_prompt,
        generation_mode=mode,
        summary=(
            f"Found {len(updates)} candidate baseline update(s) for the settlement case. "
            "Review the values and references before approving them."
            if updates
            else "No baseline updates are ready for approval from the linked evidence."
        ),
        updates=updates,
        warnings=tuple(warnings),
    )


def _match_metrics(
    indicators: list[CaseIndicator], metrics: object
) -> list[BaselineProposal]:
    if not isinstance(metrics, list):
        return []
    by_name = {item.name.casefold(): item for item in indicators}
    proposals: list[BaselineProposal] = []
    seen: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        label = str(metric.get("label", "")).strip()
        indicator = by_name.get(label.casefold())
        if indicator is None:
            indicator = _closest_indicator(indicators, label)
        if indicator is None or indicator.id in seen:
            continue
        baseline = metric.get("baseline")
        if not isinstance(baseline, (int, float)):
            continue
        refs = metric.get("evidence_refs", [])
        source_refs = tuple(str(ref).strip() for ref in refs if str(ref).strip()) if isinstance(refs, list) else ()
        rationale = str(metric.get("rationale", "Explicit value extracted from linked evidence.")).strip()
        proposals.append(
            BaselineProposal(
                indicator_id=indicator.id,
                indicator_name=indicator.name,
                baseline_value=float(baseline),
                current_value=None,
                uncertainty=None,
                quality_status="trusted" if source_refs else "partial",
                source_refs=source_refs,
                rationale=rationale or "Explicit value extracted from linked evidence.",
            )
        )
        seen.add(indicator.id)
    return proposals[:12]


def _closest_indicator(indicators: list[CaseIndicator], label: str) -> CaseIndicator | None:
    label_words = {word for word in label.casefold().replace("-", " ").split() if len(word) > 3}
    if not label_words:
        return None
    best: tuple[float, CaseIndicator] | None = None
    for indicator in indicators:
        indicator_words = {word for word in indicator.name.casefold().replace("-", " ").split() if len(word) > 3}
        score = len(label_words & indicator_words) / max(len(label_words), len(indicator_words), 1)
        if score >= 0.5 and (best is None or score > best[0]):
            best = (score, indicator)
    return best[1] if best else None
