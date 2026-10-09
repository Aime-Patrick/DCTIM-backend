from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CaseIntervention:
    id: str
    case_id: str
    workspace_id: str
    name: str
    description: str
    intervention_type: str
    priority: str
    status: str
    rationale: str
    expected_impact: str
    timeframe: str
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
