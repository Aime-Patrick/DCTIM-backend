from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CaseStatus = Literal["draft", "active", "monitoring", "completed", "archived"]


class CaseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    problem_statement: str = Field(min_length=1, max_length=5000)
    desired_outcome: str = Field(min_length=1, max_length=5000)
    territory: str | None = Field(default=None, max_length=200)
    population: str | None = Field(default=None, max_length=500)
    time_horizon: str | None = Field(default=None, max_length=200)
    decision_authority: str | None = Field(default=None, max_length=300)
    status: CaseStatus = "draft"
    success_criteria: list[str] = Field(default_factory=list, max_length=30)
    indicators: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    constraints: list[str] = Field(default_factory=list, max_length=50)
    metadata: dict[str, Any] = Field(default_factory=dict)
    review_date: datetime | None = None

    @model_validator(mode="after")
    def _validate_json_fields(self) -> CaseCreateRequest:
        _validate_json(self.indicators, "indicators")
        _validate_json(self.metadata, "metadata")
        return self


class CaseUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=200)
    problem_statement: str | None = Field(default=None, min_length=1, max_length=5000)
    desired_outcome: str | None = Field(default=None, min_length=1, max_length=5000)
    territory: str | None = Field(default=None, max_length=200)
    population: str | None = Field(default=None, max_length=500)
    time_horizon: str | None = Field(default=None, max_length=200)
    decision_authority: str | None = Field(default=None, max_length=300)
    status: CaseStatus | None = None
    success_criteria: list[str] | None = Field(default=None, max_length=30)
    indicators: list[dict[str, Any]] | None = Field(default=None, max_length=50)
    constraints: list[str] | None = Field(default=None, max_length=50)
    metadata: dict[str, Any] | None = None
    review_date: datetime | None = None

    @model_validator(mode="after")
    def _validate_update(self) -> CaseUpdateRequest:
        if not self.model_fields_set:
            raise ValueError("at least one case field is required")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("case update fields must not be null")
        if self.indicators is not None:
            _validate_json(self.indicators, "indicators")
        if self.metadata is not None:
            _validate_json(self.metadata, "metadata")
        return self


class CaseSourceLinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(min_length=1, max_length=64)
    source_role: str = Field(default="evidence", min_length=1, max_length=80)


class CaseSourceResponse(BaseModel):
    id: str
    case_id: str
    document_id: str
    source_role: str
    linked_at: datetime


class CaseSummary(BaseModel):
    id: str
    title: str
    territory: str | None
    status: CaseStatus
    revision: int
    source_count: int
    created_at: datetime
    updated_at: datetime


class CaseDetail(CaseSummary):
    workspace_id: str
    problem_statement: str
    desired_outcome: str
    population: str | None
    time_horizon: str | None
    decision_authority: str | None
    success_criteria: list[str]
    indicators: list[dict[str, Any]]
    constraints: list[str]
    metadata: dict[str, Any]
    review_date: datetime | None
    sources: list[CaseSourceResponse]


class CaseListResponse(BaseModel):
    cases: list[CaseSummary] = Field(default_factory=list)
    total: int


class DiagnosisIndicator(BaseModel):
    id: str
    name: str
    unit: str
    direction: str
    baseline_value: float | None
    target_value: float | None
    current_value: float | None
    uncertainty: float | None
    quality_status: str
    source_refs: list[str]


class DiagnosisResponse(BaseModel):
    case_id: str
    generated_at: datetime
    evidence_status: Literal["sufficient", "partial", "insufficient"]
    summary: str
    baseline_complete: bool
    indicators: list[DiagnosisIndicator]
    evidence_source_ids: list[str]
    evidence_gaps: list[str]
    assumptions: list[str]
    next_steps: list[str]


class IndicatorProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=4000)


class IndicatorProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=160)
    definition: str = Field(min_length=1, max_length=1000)
    unit: str = Field(min_length=1, max_length=80)
    direction: Literal["increase", "decrease", "neutral"] = "neutral"
    baseline_value: float | None = None
    target_value: float | None = None
    source_refs: list[str] = Field(default_factory=list, max_length=50)
    rationale: str = Field(min_length=1, max_length=2000)


class IndicatorProposalResponse(BaseModel):
    proposal_id: str
    case_id: str
    prompt: str
    generation_mode: Literal["model_assisted", "settlement_catalog_fallback"]
    summary: str
    indicators: list[IndicatorProposalItem] = Field(default_factory=list, max_length=12)
    evidence_source_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IndicatorProposalApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    proposal_id: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=4000)
    generation_mode: Literal["model_assisted", "settlement_catalog_fallback"]
    indicators: list[IndicatorProposalItem] = Field(min_length=1, max_length=12)


class IndicatorProposalApprovalResponse(BaseModel):
    proposal_id: str
    created: list[dict[str, Any]] = Field(default_factory=list)


class BaselineProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=4000)


class BaselineProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    indicator_id: str = Field(min_length=1, max_length=64)
    indicator_name: str = Field(min_length=1, max_length=160)
    baseline_value: float
    current_value: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    quality_status: Literal["partial", "trusted"]
    source_refs: list[str] = Field(default_factory=list, max_length=50)
    rationale: str = Field(min_length=1, max_length=2000)


class BaselineProposalResponse(BaseModel):
    proposal_id: str
    case_id: str
    prompt: str
    generation_mode: Literal["model_assisted", "evidence_not_found"]
    summary: str
    updates: list[BaselineProposalItem] = Field(default_factory=list, max_length=12)
    warnings: list[str] = Field(default_factory=list)


class BaselineProposalApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    proposal_id: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=4000)
    generation_mode: Literal["model_assisted", "evidence_not_found"]
    updates: list[BaselineProposalItem] = Field(min_length=1, max_length=12)


class BaselineProposalApprovalResponse(BaseModel):
    proposal_id: str
    updated: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=4000)


class ScenarioMetricResponse(BaseModel):
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
    source_refs: list[str] = Field(default_factory=list)


class ScenarioOptionResponse(BaseModel):
    key: str
    name: str
    service_access: str
    sprawl: str
    environmental_risk: str
    quantification_status: str
    rationale: str


class ScenarioComparisonResponse(BaseModel):
    case_id: str
    prompt: str
    generated_at: datetime
    evidence_status: Literal["ready", "partial", "insufficient"]
    recommendation: str
    metrics: list[ScenarioMetricResponse] = Field(default_factory=list)
    options: list[ScenarioOptionResponse] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    approved_intervention_count: int = Field(ge=0)


PlanStatus = Literal["draft", "approved"]


class PlanAction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    owner: str = Field(default="To be assigned", max_length=200)
    timeframe: str = Field(default="To be scheduled", max_length=200)
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    description: str = Field(min_length=1, max_length=2000)
    source_intervention_id: str | None = Field(default=None, max_length=64)


class PlanIndicator(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    indicator_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=160)
    unit: str = Field(min_length=1, max_length=80)
    baseline_value: float | None = None
    target_value: float | None = None
    current_value: float | None = None
    direction: str = Field(min_length=1, max_length=20)


class ImplementationPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=3000)
    decision: str = Field(min_length=1, max_length=1000)
    actions: list[PlanAction] = Field(default_factory=list, max_length=30)
    indicators: list[PlanIndicator] = Field(default_factory=list, max_length=30)
    timeline: str = Field(default="To be scheduled", max_length=500)
    notes: str = Field(default="", max_length=3000)


class ImplementationPlanResponse(ImplementationPlanRequest):
    case_id: str
    status: PlanStatus
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    approved_at: datetime | None = None
    approved_by: str | None = None


class PlanGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: str = Field(default="Decision to be confirmed after scenario comparison.", max_length=1000)
    timeline: str = Field(default="To be scheduled", max_length=500)
    notes: str = Field(default="", max_length=3000)


class MonitoringIndicatorResponse(BaseModel):
    indicator_id: str
    name: str
    unit: str
    direction: str
    baseline_value: float | None
    target_value: float | None
    current_value: float | None
    progress_percent: float | None
    status: Literal["not_started", "in_progress", "on_target", "off_track", "no_target"]
    measurement_date: datetime | None
    source_refs: list[str] = Field(default_factory=list)


class MonitoringResponse(BaseModel):
    case_id: str
    generated_at: datetime
    overall_status: Literal["not_started", "in_progress", "on_track", "off_track", "no_indicators"]
    indicators: list[MonitoringIndicatorResponse] = Field(default_factory=list)


def _validate_json(value: object, field_name: str) -> None:
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    if len(encoded) > 1_000_000:
        raise ValueError(f"{field_name} must be at most 1 MB")
