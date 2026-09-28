from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PolicyStatus = Literal[
    "created", "approved", "in_implementation", "monitoring", "completed", "on_hold"
]


class PolicyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=1_000_000)
    category: str = Field(min_length=1, max_length=100)
    status: PolicyStatus = "created"
    analysis_trace_id: str | None = Field(default=None, max_length=128)
    analysis_provider: str | None = Field(default=None, max_length=100)
    analysis: dict[str, Any] | None = None

    @field_validator("analysis")
    @classmethod
    def _validate_analysis(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            encoded = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("analysis must be JSON serializable") from exc
        if len(encoded) > 1_000_000:
            raise ValueError("analysis must be at most 1 MB")
        return value


class PolicyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, min_length=1, max_length=1_000_000)
    category: str | None = Field(default=None, min_length=1, max_length=100)
    status: PolicyStatus | None = None
    analysis: dict[str, Any] | None = None

    @field_validator("analysis")
    @classmethod
    def _validate_analysis(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            encoded = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("analysis must be JSON serializable") from exc
        if len(encoded) > 1_000_000:
            raise ValueError("analysis must be at most 1 MB")
        return value

    @model_validator(mode="after")
    def _has_update(self) -> PolicyUpdateRequest:
        if not self.model_fields_set:
            raise ValueError("at least one of title, content, category, status, or analysis is required")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("title, content, category, status, and analysis must not be null")
        return self


class PolicySummary(BaseModel):
    id: str
    title: str
    category: str
    status: PolicyStatus = "created"
    analysis_trace_id: str | None = None
    analysis_provider: str | None = None
    has_analysis: bool = False
    revision: int
    created_at: datetime
    updated_at: datetime


class PolicyDetail(PolicySummary):
    content: str
    analysis: dict[str, Any] | None = None


class PolicyListResponse(BaseModel):
    policies: list[PolicySummary] = Field(default_factory=list)
    total: int


PolicyResponse = PolicyDetail
PolicyPatchRequest = PolicyUpdateRequest
PolicyCreateResponse = PolicyDetail
