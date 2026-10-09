from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .domain import CaseIntervention


@dataclass(frozen=True)
class InterventionCreateCommand:
    name: str
    description: str
    intervention_type: str
    priority: str = "medium"
    status: str = "proposed"
    rationale: str = ""
    expected_impact: str = ""
    timeframe: str = ""
    evidence_refs: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class InterventionUpdateCommand:
    name: str | None = None
    description: str | None = None
    intervention_type: str | None = None
    priority: str | None = None
    status: str | None = None
    rationale: str | None = None
    expected_impact: str | None = None
    timeframe: str | None = None
    evidence_refs: tuple[str, ...] | None = None
    assumptions: tuple[str, ...] | None = None
    metadata: dict[str, Any] | None = None


class InterventionStore(Protocol):
    def create_intervention(
        self, workspace_id: str, case_id: str, command: InterventionCreateCommand
    ) -> CaseIntervention | None: ...

    def list_interventions(self, workspace_id: str, case_id: str) -> list[CaseIntervention]: ...

    def get_intervention(
        self, workspace_id: str, case_id: str, intervention_id: str
    ) -> CaseIntervention | None: ...

    def update_intervention(
        self,
        workspace_id: str,
        case_id: str,
        intervention_id: str,
        changes: dict[str, Any],
    ) -> CaseIntervention | None: ...


class InterventionService:
    def __init__(self, store: InterventionStore) -> None:
        self._store = store

    def create_intervention(
        self, workspace_id: str, case_id: str, command: InterventionCreateCommand
    ) -> CaseIntervention | None:
        return self._store.create_intervention(
            workspace_id,
            case_id,
            InterventionCreateCommand(
                name=_required_text(command.name, "name"),
                description=_required_text(command.description, "description"),
                intervention_type=_required_text(command.intervention_type, "intervention_type"),
                priority=_required_text(command.priority, "priority"),
                status=_required_text(command.status, "status"),
                rationale=command.rationale.strip(),
                expected_impact=command.expected_impact.strip(),
                timeframe=command.timeframe.strip(),
                evidence_refs=_clean_text(command.evidence_refs),
                assumptions=_clean_text(command.assumptions),
                metadata=dict(command.metadata or {}),
            ),
        )

    def list_interventions(self, workspace_id: str, case_id: str) -> list[CaseIntervention]:
        return self._store.list_interventions(workspace_id, case_id)

    def get_intervention(
        self, workspace_id: str, case_id: str, intervention_id: str
    ) -> CaseIntervention | None:
        return self._store.get_intervention(workspace_id, case_id, intervention_id)

    def update_intervention(
        self,
        workspace_id: str,
        case_id: str,
        intervention_id: str,
        command: InterventionUpdateCommand,
    ) -> CaseIntervention | None:
        changes: dict[str, Any] = {}
        for field_name in (
            "name", "description", "intervention_type", "priority", "status",
            "rationale", "expected_impact", "timeframe", "evidence_refs", "assumptions", "metadata",
        ):
            value = getattr(command, field_name)
            if value is None:
                continue
            if field_name in {"name", "description", "intervention_type", "priority", "status"}:
                changes[field_name] = _required_text(value, field_name)
            elif field_name in {"rationale", "expected_impact", "timeframe"}:
                changes[field_name] = value.strip()
            elif field_name in {"evidence_refs", "assumptions"}:
                changes[field_name] = _clean_text(value)
            else:
                changes[field_name] = dict(value)
        if not changes:
            return self.get_intervention(workspace_id, case_id, intervention_id)
        return self._store.update_intervention(workspace_id, case_id, intervention_id, changes)


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _clean_text(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value.strip())
