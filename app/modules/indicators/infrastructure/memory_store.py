from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ..application import IndicatorCreateCommand
from ..domain import CaseIndicator


class InMemoryIndicatorStore:
    def __init__(self) -> None:
        self._indicators: dict[str, CaseIndicator] = {}

    def create_indicator(
        self, workspace_id: str, case_id: str, command: IndicatorCreateCommand
    ) -> CaseIndicator:
        if any(
            item.workspace_id == workspace_id
            and item.case_id == case_id
            and item.name == command.name
            for item in self._indicators.values()
        ):
            raise ValueError("indicator name is already used in this case")
        now = datetime.now(timezone.utc)
        indicator = CaseIndicator(
            id=uuid4().hex,
            case_id=case_id,
            workspace_id=workspace_id,
            name=command.name,
            definition=command.definition,
            unit=command.unit,
            direction=command.direction,
            baseline_value=command.baseline_value,
            target_value=command.target_value,
            current_value=command.current_value,
            uncertainty=command.uncertainty,
            quality_status=command.quality_status,
            source_refs=command.source_refs,
            measurement_date=command.measurement_date,
            metadata=dict(command.metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._indicators[indicator.id] = indicator
        return indicator

    def list_indicators(self, workspace_id: str, case_id: str) -> list[CaseIndicator]:
        return sorted(
            (
                item
                for item in self._indicators.values()
                if item.workspace_id == workspace_id and item.case_id == case_id
            ),
            key=lambda item: (item.name, item.id),
        )

    def get_indicator(
        self, workspace_id: str, case_id: str, indicator_id: str
    ) -> CaseIndicator | None:
        item = self._indicators.get(indicator_id)
        if item is None or item.workspace_id != workspace_id or item.case_id != case_id:
            return None
        return item

    def update_indicator(
        self,
        workspace_id: str,
        case_id: str,
        indicator_id: str,
        changes: dict[str, Any],
    ) -> CaseIndicator | None:
        item = self.get_indicator(workspace_id, case_id, indicator_id)
        if item is None:
            return None
        if "name" in changes and any(
            other.id != item.id
            and other.workspace_id == workspace_id
            and other.case_id == case_id
            and other.name == changes["name"]
            for other in self._indicators.values()
        ):
            raise ValueError("indicator name is already used in this case")
        updated = replace(item, **changes, updated_at=datetime.now(timezone.utc))
        self._indicators[indicator_id] = updated
        return updated
