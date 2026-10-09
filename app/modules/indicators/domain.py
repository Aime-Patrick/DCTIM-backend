from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CaseIndicator:
    id: str
    case_id: str
    workspace_id: str
    name: str
    definition: str
    unit: str
    direction: str
    baseline_value: float | None
    target_value: float | None
    current_value: float | None
    uncertainty: float | None
    quality_status: str
    source_refs: tuple[str, ...]
    measurement_date: datetime | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
