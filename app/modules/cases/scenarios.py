from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from ..indicators.domain import CaseIndicator
from ..interventions.domain import CaseIntervention
from .domain import TransformationCase


ScenarioReadiness = Literal["ready", "partial", "insufficient"]


@dataclass(frozen=True)
class ScenarioMetric:
    indicator_id: str
    name: str
    unit: str
    direction: str
    baseline_value: float | None
    target_value: float | None
    current_value: float | None
    target_gap: float | None
    baseline_to_target_gap: float | None
    progress_percent: float | None
    equation: str
    status: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioOption:
    key: str
    name: str
    service_access: str
    sprawl: str
    environmental_risk: str
    quantification_status: str
    rationale: str


@dataclass(frozen=True)
class ScenarioComparison:
    case_id: str
    prompt: str
    generated_at: datetime
    evidence_status: ScenarioReadiness
    recommendation: str
    metrics: tuple[ScenarioMetric, ...]
    options: tuple[ScenarioOption, ...]
    equations: tuple[str, ...]
    assumptions: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    approved_intervention_count: int


def build_scenario_comparison(
    case: TransformationCase,
    indicators: list[CaseIndicator],
    interventions: list[CaseIntervention],
    prompt: str,
) -> ScenarioComparison:
    metrics = tuple(_metric_from_indicator(item) for item in indicators)
    complete = [item for item in metrics if item.baseline_value is not None and item.target_value is not None]
    referenced = [item for item in complete if item.source_refs]
    if not indicators:
        evidence_status: ScenarioReadiness = "insufficient"
    elif len(complete) == len(indicators) and len(referenced) == len(complete):
        evidence_status = "ready"
    else:
        evidence_status = "partial"

    evidence_gaps = list(_evidence_gaps(metrics))
    assumptions = [
        "Scenario-specific service, population, cost, and environmental inputs were not supplied; those outcomes are not numerically projected.",
        "The comparison uses approved case indicators as the measurable baseline and target frame.",
    ]
    if not interventions:
        assumptions.append("No approved intervention is attached to this case yet.")

    normalized = prompt.casefold()
    compares_settlements = "rurban" in normalized and ("consolidat" in normalized or "agglomeration" in normalized)
    recommendation = (
        "Carry consolidated settlements forward as the leading scenario for review, while quantifying service capacity and environmental exposure before approval."
        if compares_settlements
        else "Use the measurable indicator gaps to refine the scenario assumptions before selecting a preferred option."
    )
    options = (
        ScenarioOption(
            key="rurban_expansion",
            name="Rurban expansion",
            service_access="Not quantified — add population and service-capacity assumptions.",
            sprawl="Higher pressure if growth continues outside planned boundaries.",
            environmental_risk="Not quantified — add settlement exposure and land-conversion data.",
            quantification_status="qualitative_only",
            rationale="The scenario represents continued dispersed settlement growth; no numeric service or land-use lever was supplied.",
        ),
        ScenarioOption(
            key="consolidated_settlements",
            name="Consolidated settlements",
            service_access="Not quantified — add population, capacity, and investment assumptions.",
            sprawl="Lower pressure if consolidation and boundary controls are enforced.",
            environmental_risk="Not quantified — add site exposure and mitigation assumptions.",
            quantification_status="qualitative_only",
            rationale="The scenario represents planned concentration; the case still needs explicit capacity and risk parameters for numeric projections.",
        ),
    )
    return ScenarioComparison(
        case_id=case.id,
        prompt=prompt,
        generated_at=datetime.now(timezone.utc),
        evidence_status=evidence_status,
        recommendation=recommendation,
        metrics=metrics,
        options=options,
        equations=(
            "Target gap = target − current for increase indicators; current − target for decrease indicators.",
            "Required change = target − baseline for increase indicators; baseline − target for decrease indicators.",
            "Progress = (current − baseline) ÷ (target − baseline) × 100, direction-adjusted when a current value exists.",
        ),
        assumptions=tuple(assumptions),
        evidence_gaps=tuple(evidence_gaps),
        approved_intervention_count=len([item for item in interventions if item.status == "approved"]),
    )


def _metric_from_indicator(indicator: CaseIndicator) -> ScenarioMetric:
    baseline = indicator.baseline_value
    target = indicator.target_value
    current = indicator.current_value
    direction = indicator.direction
    required_change = None
    target_gap = None
    progress = None
    equation = "Awaiting baseline and target values."
    status = "not_ready"

    if baseline is not None and target is not None:
        if direction == "decrease":
            required_change = baseline - target
            equation = "required change = baseline − target"
            if current is not None:
                target_gap = current - target
                denominator = baseline - target
                progress = ((baseline - current) / denominator * 100) if denominator else None
        else:
            required_change = target - baseline
            equation = "required change = target − baseline"
            if current is not None:
                target_gap = target - current
                denominator = target - baseline
                progress = ((current - baseline) / denominator * 100) if denominator else None
        status = "ready" if current is not None else "current_value_missing"

    return ScenarioMetric(
        indicator_id=indicator.id,
        name=indicator.name,
        unit=indicator.unit,
        direction=direction,
        baseline_value=baseline,
        target_value=target,
        current_value=current,
        target_gap=target_gap,
        baseline_to_target_gap=required_change,
        progress_percent=round(progress, 2) if progress is not None else None,
        equation=equation,
        status=status,
        source_refs=indicator.source_refs,
    )


def _evidence_gaps(metrics: tuple[ScenarioMetric, ...]) -> tuple[str, ...]:
    gaps: list[str] = []
    for metric in metrics:
        if metric.baseline_value is None:
            gaps.append(f"{metric.name}: baseline value is missing.")
        if metric.target_value is None:
            gaps.append(f"{metric.name}: target value is missing.")
        if not metric.source_refs:
            gaps.append(f"{metric.name}: source reference is missing.")
    return tuple(gaps)
