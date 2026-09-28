from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PolicyArtifact:
    id: str
    workspace_id: str
    title: str
    content: str
    category: str
    status: str
    analysis_trace_id: str | None
    analysis_provider: str | None
    analysis: dict[str, Any] | None
    revision: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PolicySummary:
    id: str
    title: str
    category: str
    status: str
    analysis_trace_id: str | None
    analysis_provider: str | None
    has_analysis: bool
    revision: int
    created_at: datetime
    updated_at: datetime
