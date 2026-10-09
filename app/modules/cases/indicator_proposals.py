from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..rag.application import RagService
from .domain import TransformationCase


@dataclass(frozen=True)
class IndicatorProposal:
    name: str
    definition: str
    unit: str
    direction: str
    baseline_value: float | None
    target_value: float | None
    source_refs: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class IndicatorProposalPackage:
    proposal_id: str
    case_id: str
    prompt: str
    generation_mode: str
    summary: str
    indicators: tuple[IndicatorProposal, ...]
    evidence_source_ids: tuple[str, ...]
    warnings: tuple[str, ...]


_SETTLEMENT_CATALOG: tuple[IndicatorProposal, ...] = (
    IndicatorProposal(
        name="Population within 30 minutes of basic services",
        definition="Share of the settlement population within a 30-minute travel threshold of agreed basic services.",
        unit="percent",
        direction="increase",
        baseline_value=None,
        target_value=None,
        source_refs=(),
        rationale="Measures whether settlement growth improves practical access to services.",
    ),
    IndicatorProposal(
        name="Basic infrastructure coverage",
        definition="Share of households or developed plots covered by the agreed basic infrastructure package.",
        unit="percent",
        direction="increase",
        baseline_value=None,
        target_value=None,
        source_refs=(),
        rationale="Tracks whether settlement expansion is matched by infrastructure provision.",
    ),
    IndicatorProposal(
        name="Settlement built-up area",
        definition="Total mapped built-up area inside the agreed settlement study boundary.",
        unit="square kilometres",
        direction="neutral",
        baseline_value=None,
        target_value=None,
        source_refs=(),
        rationale="Provides the spatial growth baseline needed to compare settlement scenarios.",
    ),
    IndicatorProposal(
        name="Settlement growth outside planned boundaries",
        definition="Share of new settlement growth occurring outside approved or planned settlement boundaries.",
        unit="percent",
        direction="decrease",
        baseline_value=None,
        target_value=None,
        source_refs=(),
        rationale="Tests whether growth follows the plan’s boundary-control and compact-development intent.",
    ),
    IndicatorProposal(
        name="Land converted to settlement use",
        definition="Annual area converted from non-settlement land uses to settlement use within the study area.",
        unit="hectares per year",
        direction="decrease",
        baseline_value=None,
        target_value=None,
        source_refs=(),
        rationale="Makes land-use pressure visible and supports comparison of expansion patterns.",
    ),
    IndicatorProposal(
        name="Settlement area exposed to environmental risk",
        definition="Share of settlement land or population located in mapped environmental or hazard-risk areas.",
        unit="percent",
        direction="decrease",
        baseline_value=None,
        target_value=None,
        source_refs=(),
        rationale="Keeps environmental and hazard exposure in the settlement decision frame.",
    ),
)


def build_indicator_proposal(
    case: TransformationCase,
    prompt: str,
    rag: RagService | None = None,
) -> IndicatorProposalPackage:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("prompt is required")

    analysis: dict[str, Any] = {}
    if rag is not None and case.sources:
        query = (
            "Prepare a source-grounded settlement indicator proposal for this DC-TIM case. "
            "Return metrics that can be used as indicators; do not invent values. "
            f"Case problem: {case.problem_statement}\n"
            f"Desired outcome: {case.desired_outcome}\n"
            f"User request: {normalized_prompt}"
        )
        try:
            analysis = rag.analyze(
                case.workspace_id,
                query,
                top_k=8,
                document_ids=tuple(source.document_id for source in case.sources),
            ).analysis
        except Exception:
            # Proposal generation must degrade to the controlled domain catalog when
            # a provider or vector store is unavailable.
            analysis = {}

    model_indicators = _from_analysis(analysis)
    selected = _merge_unique(model_indicators, _select_catalog(normalized_prompt))
    mode = "model_assisted" if model_indicators else "settlement_catalog_fallback"
    warnings: list[str] = []
    if not case.sources:
        warnings.append("Link an evidence source before approving this proposal.")
    if not any(item.source_refs for item in selected):
        warnings.append("The proposed indicators still need exact page, table, map, or dataset references.")
    if not any(item.baseline_value is not None for item in selected):
        warnings.append("No baseline values were found; add a measured dataset before scenario comparison.")

    return IndicatorProposalPackage(
        proposal_id=uuid4().hex,
        case_id=case.id,
        prompt=normalized_prompt,
        generation_mode=mode,
        summary=(
            f"Prepared {len(selected)} settlement indicator proposal(s) for {case.title}. "
            "Review the definitions and approve only the indicators that fit the decision."
        ),
        indicators=tuple(selected),
        evidence_source_ids=tuple(source.document_id for source in case.sources),
        warnings=tuple(warnings),
    )


def _from_analysis(analysis: dict[str, Any]) -> list[IndicatorProposal]:
    proposals: list[IndicatorProposal] = []
    metrics = analysis.get("metrics")
    if not isinstance(metrics, list):
        return proposals
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        label = str(metric.get("label", "")).strip()
        unit = str(metric.get("unit", "")).strip()
        direction = str(metric.get("direction", "neutral")).strip()
        if not label or not unit or direction not in {"increase", "decrease", "neutral"}:
            continue
        refs = metric.get("evidence_refs", [])
        source_refs = tuple(str(ref).strip() for ref in refs if str(ref).strip()) if isinstance(refs, list) else ()
        baseline = metric.get("baseline")
        baseline_value = float(baseline) if isinstance(baseline, (int, float)) else None
        rationale = str(metric.get("rationale", "Suggested from the linked evidence.")).strip()
        proposals.append(
            IndicatorProposal(
                name=label,
                definition=f"Measured value for {label.lower()} in the settlement study area.",
                unit=unit,
                direction=direction,
                baseline_value=baseline_value,
                target_value=None,
                source_refs=source_refs,
                rationale=rationale or "Suggested from the linked evidence.",
            )
        )
    return proposals[:8]


def _select_catalog(prompt: str) -> list[IndicatorProposal]:
    lowered = prompt.lower()
    keyword_groups = {
        "service": ("service", "access", "school", "health", "water", "facility"),
        "infrastructure": ("infrastructure", "road", "utility", "coverage"),
        "growth": ("growth", "built-up", "built up", "density", "expansion", "boundary"),
        "land": ("land", "conversion", "agriculture", "sprawl"),
        "risk": ("risk", "environment", "hazard", "flood", "sensitive"),
    }
    if any(word in lowered for word in ("all", "complete", "full", "indicators")):
        return list(_SETTLEMENT_CATALOG)
    selected: list[IndicatorProposal] = []
    for item in _SETTLEMENT_CATALOG:
        haystack = f"{item.name} {item.definition} {item.rationale}".lower()
        if any(keyword in lowered and keyword in haystack for group in keyword_groups.values() for keyword in group):
            selected.append(item)
    return selected or list(_SETTLEMENT_CATALOG)


def _merge_unique(*groups: list[IndicatorProposal]) -> list[IndicatorProposal]:
    merged: list[IndicatorProposal] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = item.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged[:12]
