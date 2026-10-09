from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..indicators.domain import CaseIndicator
from .domain import TransformationCase


@dataclass(frozen=True)
class DiagnosisPackage:
    case_id: str
    generated_at: datetime
    evidence_status: str
    summary: str
    baseline_complete: bool
    indicators: tuple[CaseIndicator, ...]
    evidence_source_ids: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    assumptions: tuple[str, ...]
    next_steps: tuple[str, ...]


def build_diagnosis(
    case: TransformationCase, indicators: list[CaseIndicator]
) -> DiagnosisPackage:
    gaps: list[str] = []
    if not case.territory:
        gaps.append("The pilot territory has not been defined.")
    if not case.sources:
        gaps.append("No evidence sources are linked to the case.")
    if not indicators:
        gaps.append("No structured settlement indicators have been configured.")

    incomplete = [
        indicator.name
        for indicator in indicators
        if indicator.baseline_value is None and indicator.current_value is None
    ]
    if incomplete:
        gaps.append(f"Baseline or current values are missing for: {', '.join(incomplete)}.")

    unreferenced = [indicator.name for indicator in indicators if not indicator.source_refs]
    if unreferenced:
        gaps.append(f"Indicators lack source references: {', '.join(unreferenced)}.")

    baseline_complete = bool(indicators) and not incomplete
    if not case.sources or not indicators:
        evidence_status = "insufficient"
    elif gaps or any(item.quality_status != "trusted" for item in indicators):
        evidence_status = "partial"
    else:
        evidence_status = "sufficient"

    summary = (
        f"{case.title} currently has {len(indicators)} configured indicator(s) and "
        f"{len(case.sources)} linked evidence source(s)."
    )
    next_steps = [
        "Confirm the pilot territory and decision authority.",
        "Link each indicator to a source and assess its quality.",
        "Complete baseline and target values before comparing interventions.",
    ]
    return DiagnosisPackage(
        case_id=case.id,
        generated_at=datetime.now(timezone.utc),
        evidence_status=evidence_status,
        summary=summary,
        baseline_complete=baseline_complete,
        indicators=tuple(indicators),
        evidence_source_ids=tuple(source.document_id for source in case.sources),
        evidence_gaps=tuple(gaps),
        assumptions=("This diagnosis is a structured evidence-readiness assessment, not a causal finding.",),
        next_steps=tuple(next_steps),
    )
