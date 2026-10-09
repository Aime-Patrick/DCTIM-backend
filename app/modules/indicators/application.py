from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .domain import CaseIndicator


@dataclass(frozen=True)
class IndicatorCreateCommand:
    name: str
    definition: str
    unit: str
    direction: str = "neutral"
    baseline_value: float | None = None
    target_value: float | None = None
    current_value: float | None = None
    uncertainty: float | None = None
    quality_status: str = "unassessed"
    source_refs: tuple[str, ...] = ()
    measurement_date: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class IndicatorUpdateCommand:
    name: str | None = None
    definition: str | None = None
    unit: str | None = None
    direction: str | None = None
    baseline_value: float | None = None
    target_value: float | None = None
    current_value: float | None = None
    uncertainty: float | None = None
    quality_status: str | None = None
    source_refs: tuple[str, ...] | None = None
    measurement_date: datetime | None = None
    metadata: dict[str, Any] | None = None


class IndicatorStore(Protocol):
    def create_indicator(
        self, workspace_id: str, case_id: str, command: IndicatorCreateCommand
    ) -> CaseIndicator | None: ...

    def list_indicators(self, workspace_id: str, case_id: str) -> list[CaseIndicator]: ...

    def get_indicator(
        self, workspace_id: str, case_id: str, indicator_id: str
    ) -> CaseIndicator | None: ...

    def update_indicator(
        self,
        workspace_id: str,
        case_id: str,
        indicator_id: str,
        changes: dict[str, Any],
    ) -> CaseIndicator | None: ...


class IndicatorService:
    def __init__(self, store: IndicatorStore) -> None:
        self._store = store

    def create_indicator(
        self, workspace_id: str, case_id: str, command: IndicatorCreateCommand
    ) -> CaseIndicator | None:
        return self._store.create_indicator(
            workspace_id,
            case_id,
            IndicatorCreateCommand(
                name=_required_text(command.name, "name"),
                definition=_required_text(command.definition, "definition"),
                unit=_required_text(command.unit, "unit"),
                direction=_required_text(command.direction, "direction"),
                baseline_value=command.baseline_value,
                target_value=command.target_value,
                current_value=command.current_value,
                uncertainty=command.uncertainty,
                quality_status=_required_text(command.quality_status, "quality_status"),
                source_refs=_clean_refs(command.source_refs),
                measurement_date=command.measurement_date,
                metadata=dict(command.metadata or {}),
            ),
        )

    def list_indicators(self, workspace_id: str, case_id: str) -> list[CaseIndicator]:
        return self._store.list_indicators(workspace_id, case_id)

    def get_indicator(
        self, workspace_id: str, case_id: str, indicator_id: str
    ) -> CaseIndicator | None:
        return self._store.get_indicator(workspace_id, case_id, indicator_id)

    def update_indicator(
        self,
        workspace_id: str,
        case_id: str,
        indicator_id: str,
        command: IndicatorUpdateCommand,
    ) -> CaseIndicator | None:
        changes: dict[str, Any] = {}
        for field_name in (
            "name", "definition", "unit", "direction", "baseline_value", "target_value",
            "current_value", "uncertainty", "quality_status", "source_refs",
            "measurement_date", "metadata",
        ):
            value = getattr(command, field_name)
            if value is None:
                continue
            if field_name in {"name", "definition", "unit", "direction", "quality_status"}:
                changes[field_name] = _required_text(value, field_name)
            elif field_name == "source_refs":
                changes[field_name] = _clean_refs(value)
            elif field_name == "metadata":
                changes[field_name] = dict(value)
            else:
                changes[field_name] = value
        if not changes:
            return self.get_indicator(workspace_id, case_id, indicator_id)
        return self._store.update_indicator(workspace_id, case_id, indicator_id, changes)


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _clean_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value.strip())
