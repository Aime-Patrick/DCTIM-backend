from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


IndicatorDirection = Literal["increase", "decrease", "neutral"]
IndicatorQuality = Literal["unassessed", "partial", "trusted", "insufficient"]


class IndicatorCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=160)
    definition: str = Field(min_length=1, max_length=1000)
    unit: str = Field(min_length=1, max_length=80)
    direction: IndicatorDirection = "neutral"
    baseline_value: float | None = None
    target_value: float | None = None
    current_value: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    quality_status: IndicatorQuality = "unassessed"
    source_refs: list[str] = Field(default_factory=list, max_length=50)
    measurement_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_json_fields(self) -> IndicatorCreateRequest:
        _validate_json(self.source_refs, "source_refs")
        _validate_json(self.metadata, "metadata")
        return self


class IndicatorUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=160)
    definition: str | None = Field(default=None, min_length=1, max_length=1000)
    unit: str | None = Field(default=None, min_length=1, max_length=80)
    direction: IndicatorDirection | None = None
    baseline_value: float | None = None
    target_value: float | None = None
    current_value: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    quality_status: IndicatorQuality | None = None
    source_refs: list[str] | None = Field(default=None, max_length=50)
    measurement_date: datetime | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_update(self) -> IndicatorUpdateRequest:
        if not self.model_fields_set:
            raise ValueError("at least one indicator field is required")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("indicator update fields must not be null")
        if self.source_refs is not None:
            _validate_json(self.source_refs, "source_refs")
        if self.metadata is not None:
            _validate_json(self.metadata, "metadata")
        return self


class IndicatorResponse(BaseModel):
    id: str
    case_id: str
    workspace_id: str
    name: str
    definition: str
    unit: str
    direction: IndicatorDirection
    baseline_value: float | None
    target_value: float | None
    current_value: float | None
    uncertainty: float | None
    quality_status: IndicatorQuality
    source_refs: list[str]
    measurement_date: datetime | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class IndicatorListResponse(BaseModel):
    indicators: list[IndicatorResponse] = Field(default_factory=list)


def _validate_json(value: object, field_name: str) -> None:
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    if len(encoded) > 1_000_000:
        raise ValueError(f"{field_name} must be at most 1 MB")
