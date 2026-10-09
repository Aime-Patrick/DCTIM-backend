from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ..application import InterventionCreateCommand
from ..domain import CaseIntervention


class InMemoryInterventionStore:
    def __init__(self) -> None:
        self._interventions: dict[str, CaseIntervention] = {}

    def create_intervention(
        self, workspace_id: str, case_id: str, command: InterventionCreateCommand
    ) -> CaseIntervention:
        if any(
            item.workspace_id == workspace_id
            and item.case_id == case_id
            and item.name.casefold() == command.name.casefold()
            for item in self._interventions.values()
        ):
            raise ValueError("intervention name is already used in this case")
        now = datetime.now(timezone.utc)
        intervention = CaseIntervention(
            id=uuid4().hex,
            case_id=case_id,
            workspace_id=workspace_id,
            name=command.name,
            description=command.description,
            intervention_type=command.intervention_type,
            priority=command.priority,
            status=command.status,
            rationale=command.rationale,
            expected_impact=command.expected_impact,
            timeframe=command.timeframe,
            evidence_refs=command.evidence_refs,
            assumptions=command.assumptions,
            metadata=dict(command.metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._interventions[intervention.id] = intervention
        return intervention

    def list_interventions(self, workspace_id: str, case_id: str) -> list[CaseIntervention]:
        return sorted(
            (
                item
                for item in self._interventions.values()
                if item.workspace_id == workspace_id and item.case_id == case_id
            ),
            key=lambda item: (item.priority, item.name.casefold(), item.id),
        )

    def get_intervention(
        self, workspace_id: str, case_id: str, intervention_id: str
    ) -> CaseIntervention | None:
        item = self._interventions.get(intervention_id)
        if item is None or item.workspace_id != workspace_id or item.case_id != case_id:
            return None
        return item

    def update_intervention(
        self,
        workspace_id: str,
        case_id: str,
        intervention_id: str,
        changes: dict[str, Any],
    ) -> CaseIntervention | None:
        item = self.get_intervention(workspace_id, case_id, intervention_id)
        if item is None:
            return None
        if "name" in changes and any(
            other.id != item.id
            and other.workspace_id == workspace_id
            and other.case_id == case_id
            and other.name.casefold() == changes["name"].casefold()
            for other in self._interventions.values()
        ):
            raise ValueError("intervention name is already used in this case")
        updated = replace(item, **changes, updated_at=datetime.now(timezone.utc))
        self._interventions[intervention_id] = updated
        return updated
