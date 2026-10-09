from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


InterventionPriority = Literal["low", "medium", "high", "critical"]
InterventionStatus = Literal["proposed", "approved", "in_progress", "completed", "rejected"]


class InterventionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    intervention_type: str = Field(min_length=1, max_length=80)
    priority: InterventionPriority = "medium"
    status: InterventionStatus = "proposed"
    rationale: str = Field(default="", max_length=2000)
    expected_impact: str = Field(default="", max_length=2000)
    timeframe: str = Field(default="", max_length=200)
    evidence_refs: list[str] = Field(default_factory=list, max_length=50)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterventionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, min_length=1, max_length=4000)
    intervention_type: str | None = Field(default=None, min_length=1, max_length=80)
    priority: InterventionPriority | None = None
    status: InterventionStatus | None = None
    rationale: str | None = Field(default=None, max_length=2000)
    expected_impact: str | None = Field(default=None, max_length=2000)
    timeframe: str | None = Field(default=None, max_length=200)
    evidence_refs: list[str] | None = Field(default=None, max_length=50)
    assumptions: list[str] | None = Field(default=None, max_length=20)
    metadata: dict[str, Any] | None = None


class InterventionResponse(BaseModel):
    id: str
    case_id: str
    workspace_id: str
    name: str
    description: str
    intervention_type: str
    priority: InterventionPriority
    status: InterventionStatus
    rationale: str
    expected_impact: str
    timeframe: str
    evidence_refs: list[str]
    assumptions: list[str]
    metadata: dict[str, Any]
    created_at: Any
    updated_at: Any


class InterventionListResponse(BaseModel):
    interventions: list[InterventionResponse] = Field(default_factory=list)


class InterventionProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=4000)


class InterventionProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    intervention_type: str = Field(min_length=1, max_length=80)
    priority: InterventionPriority = "medium"
    rationale: str = Field(min_length=1, max_length=2000)
    expected_impact: str = Field(min_length=1, max_length=2000)
    timeframe: str = Field(min_length=1, max_length=200)
    evidence_refs: list[str] = Field(default_factory=list, max_length=50)
    assumptions: list[str] = Field(default_factory=list, max_length=20)


class InterventionProposalResponse(BaseModel):
    proposal_id: str
    case_id: str
    prompt: str
    generation_mode: Literal["model_assisted", "settlement_catalog_fallback"]
    summary: str
    interventions: list[InterventionProposalItem] = Field(default_factory=list, max_length=12)
    warnings: list[str] = Field(default_factory=list)


class InterventionProposalApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    proposal_id: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=4000)
    generation_mode: Literal["model_assisted", "settlement_catalog_fallback"]
    interventions: list[InterventionProposalItem] = Field(min_length=1, max_length=12)


class InterventionProposalApprovalResponse(BaseModel):
    proposal_id: str
    created: list[InterventionResponse] = Field(default_factory=list)
