from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..cases.domain import TransformationCase
from ..rag.application import RagService


@dataclass(frozen=True)
class InterventionProposal:
    name: str
    description: str
    intervention_type: str
    priority: str
    rationale: str
    expected_impact: str
    timeframe: str
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class InterventionProposalPackage:
    proposal_id: str
    case_id: str
    prompt: str
    generation_mode: str
    summary: str
    interventions: tuple[InterventionProposal, ...]
    warnings: tuple[str, ...]


_SETTLEMENT_CATALOG: tuple[InterventionProposal, ...] = (
    InterventionProposal(
        name="Planned settlement boundary control",
        description="Define, publish, and monitor settlement boundaries so new growth is directed into planned areas.",
        intervention_type="boundary_control",
        priority="high",
        rationale="Responds to the settlement-planning need to reduce unmanaged outward expansion.",
        expected_impact="Lower share of growth outside planned boundaries and better coordination of infrastructure.",
        timeframe="0–3 years",
        evidence_refs=(),
        assumptions=("A legally or administratively recognized settlement boundary can be established.",),
    ),
    InterventionProposal(
        name="Compact settlement consolidation",
        description="Prioritize infill, consolidation, and phased development within existing settlement areas before opening new expansion areas.",
        intervention_type="settlement_consolidation",
        priority="high",
        rationale="Supports more efficient use of land and existing infrastructure capacity.",
        expected_impact="Reduced infrastructure pressure and more efficient access to services.",
        timeframe="1–5 years",
        evidence_refs=(),
        assumptions=("Suitable infill or underused land can be identified without increasing hazard exposure.",),
    ),
    InterventionProposal(
        name="Phased basic-service investment",
        description="Sequence water, sanitation, education, health, and other basic-service investments alongside settlement growth.",
        intervention_type="service_investment",
        priority="high",
        rationale="Connects settlement expansion decisions to measurable service-access improvements.",
        expected_impact="Higher population share within the agreed access threshold for basic services.",
        timeframe="1–5 years",
        evidence_refs=(),
        assumptions=("Service-capacity and location data are available for prioritization.",),
    ),
    InterventionProposal(
        name="Infrastructure-first settlement phasing",
        description="Allow new settlement phases only when minimum infrastructure and access conditions are met.",
        intervention_type="infrastructure_phasing",
        priority="medium",
        rationale="Reduces the risk that settlement growth outpaces infrastructure delivery.",
        expected_impact="Improved infrastructure coverage and fewer high-cost retrofits.",
        timeframe="1–5 years",
        evidence_refs=(),
        assumptions=("Minimum service and network thresholds can be agreed and monitored.",),
    ),
    InterventionProposal(
        name="Environmental-risk avoidance",
        description="Exclude mapped environmental and hazard-risk areas from settlement expansion and guide safer alternatives.",
        intervention_type="risk_avoidance",
        priority="high",
        rationale="Keeps environmental exposure in the settlement decision and protects sensitive areas.",
        expected_impact="Lower settlement exposure to mapped environmental or hazard risks.",
        timeframe="0–3 years",
        evidence_refs=(),
        assumptions=("Current hazard and environmentally sensitive-area maps are available.",),
    ),
)


def build_intervention_proposal(
    case: TransformationCase,
    prompt: str,
    rag: RagService | None = None,
) -> InterventionProposalPackage:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("prompt is required")

    analysis: dict[str, Any] = {}
    if rag is not None and case.sources:
        query = (
            "Prepare a source-grounded intervention proposal for this DC-TIM settlement case. "
            "Return practical interventions, cite supporting chunk IDs, and do not invent policy facts.\n"
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
            analysis = {}

    model_interventions = _from_analysis(analysis)
    selected = _merge_unique(model_interventions, _select_catalog(normalized_prompt))
    warnings: list[str] = []
    if not case.sources:
        warnings.append("Link an evidence source before approving this intervention proposal.")
    if not any(item.evidence_refs for item in selected):
        warnings.append("Interventions still need exact page, table, map, or dataset references.")

    return InterventionProposalPackage(
        proposal_id=uuid4().hex,
        case_id=case.id,
        prompt=normalized_prompt,
        generation_mode="model_assisted" if model_interventions else "settlement_catalog_fallback",
        summary=(
            f"Prepared {len(selected)} settlement intervention proposal(s) for {case.title}. "
            "Review the actions, trade-offs, and assumptions before approval."
        ),
        interventions=tuple(selected),
        warnings=tuple(warnings),
    )


def _from_analysis(analysis: dict[str, Any]) -> list[InterventionProposal]:
    proposals: list[InterventionProposal] = []
    recommendations = analysis.get("recommendations")
    if not isinstance(recommendations, list):
        return proposals
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        detail = str(item.get("detail", "")).strip()
        if not title or not detail:
            continue
        priority = str(item.get("priority", "medium")).strip()
        if priority not in {"low", "medium", "high", "critical"}:
            priority = "medium"
        refs = item.get("evidence_refs", [])
        evidence_refs = tuple(str(ref).strip() for ref in refs if str(ref).strip()) if isinstance(refs, list) else ()
        expected = str(item.get("expected_impact", "")).strip() or "Impact requires baseline and scenario assessment."
        timeframe = str(item.get("timeframe", "")).strip() or "To be determined"
        proposals.append(
            InterventionProposal(
                name=title,
                description=detail,
                intervention_type=_classify_type(f"{title} {detail}"),
                priority=priority,
                rationale="Suggested from the linked evidence.",
                expected_impact=expected,
                timeframe=timeframe,
                evidence_refs=evidence_refs,
                assumptions=("Implementation feasibility and costs require local validation.",),
            )
        )
    return proposals[:6]


def _select_catalog(prompt: str) -> list[InterventionProposal]:
    lowered = prompt.lower()
    if any(word in lowered for word in ("all", "complete", "full", "interventions")):
        return list(_SETTLEMENT_CATALOG)
    groups = {
        "boundary": ("boundary", "sprawl", "outward", "planned growth"),
        "consolidation": ("compact", "infill", "consolidat", "density"),
        "service": ("service", "access", "school", "health", "water", "facility"),
        "infrastructure": ("infrastructure", "road", "utility", "network"),
        "risk": ("risk", "environment", "hazard", "flood", "sensitive"),
    }
    selected: list[InterventionProposal] = []
    for item in _SETTLEMENT_CATALOG:
        haystack = f"{item.name} {item.description} {item.intervention_type}".lower()
        if any(keyword in lowered and keyword in haystack for keywords in groups.values() for keyword in keywords):
            selected.append(item)
    return selected or list(_SETTLEMENT_CATALOG)


def _classify_type(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("boundary", "sprawl", "compact", "infill")):
        return "settlement_form"
    if any(word in lowered for word in ("service", "school", "health", "water")):
        return "service_investment"
    if any(word in lowered for word in ("risk", "environment", "hazard")):
        return "risk_avoidance"
    if any(word in lowered for word in ("road", "infrastructure", "utility")):
        return "infrastructure_phasing"
    return "settlement_governance"


def _merge_unique(*groups: list[InterventionProposal]) -> list[InterventionProposal]:
    merged: list[InterventionProposal] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = item.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged[:12]
