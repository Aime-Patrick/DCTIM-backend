from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CaseSource:
    id: str
    case_id: str
    document_id: str
    source_role: str
    linked_at: datetime


@dataclass(frozen=True)
class TransformationCase:
    id: str
    workspace_id: str
    title: str
    problem_statement: str
    desired_outcome: str
    territory: str | None
    population: str | None
    time_horizon: str | None
    decision_authority: str | None
    status: str
    success_criteria: tuple[str, ...]
    indicators: tuple[dict[str, Any], ...]
    constraints: tuple[str, ...]
    metadata: dict[str, Any]
    revision: int
    review_date: datetime | None
    created_at: datetime
    updated_at: datetime
    sources: tuple[CaseSource, ...] = ()


@dataclass(frozen=True)
class TransformationCaseSummary:
    id: str
    title: str
    territory: str | None
    status: str
    revision: int
    source_count: int
    created_at: datetime
    updated_at: datetime
