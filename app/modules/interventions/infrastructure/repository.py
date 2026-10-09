from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..application import InterventionCreateCommand
from ..domain import CaseIntervention
from .models import CaseInterventionRow


class InterventionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_intervention(
        self, workspace_id: str, case_id: str, command: InterventionCreateCommand
    ) -> CaseIntervention:
        row = CaseInterventionRow(
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
            evidence_refs=list(command.evidence_refs),
            assumptions=list(command.assumptions),
            metadata_=dict(command.metadata or {}),
        )
        self._session.add(row)
        self._session.flush()
        return _to_domain(row)

    def list_interventions(self, workspace_id: str, case_id: str) -> list[CaseIntervention]:
        rows = self._session.scalars(
            select(CaseInterventionRow)
            .where(
                CaseInterventionRow.workspace_id == workspace_id,
                CaseInterventionRow.case_id == case_id,
            )
            .order_by(CaseInterventionRow.priority.asc(), CaseInterventionRow.name.asc())
        ).all()
        return [_to_domain(row) for row in rows]

    def get_intervention(
        self, workspace_id: str, case_id: str, intervention_id: str
    ) -> CaseIntervention | None:
        row = self._session.scalar(
            select(CaseInterventionRow).where(
                CaseInterventionRow.id == intervention_id,
                CaseInterventionRow.workspace_id == workspace_id,
                CaseInterventionRow.case_id == case_id,
            )
        )
        return _to_domain(row) if row is not None else None

    def update_intervention(
        self,
        workspace_id: str,
        case_id: str,
        intervention_id: str,
        changes: dict[str, Any],
    ) -> CaseIntervention | None:
        row = self._session.scalar(
            select(CaseInterventionRow).where(
                CaseInterventionRow.id == intervention_id,
                CaseInterventionRow.workspace_id == workspace_id,
                CaseInterventionRow.case_id == case_id,
            )
        )
        if row is None:
            return None
        for field_name in (
            "name", "description", "intervention_type", "priority", "status",
            "rationale", "expected_impact", "timeframe", "evidence_refs", "assumptions",
        ):
            if field_name in changes:
                setattr(row, field_name, changes[field_name])
        if "metadata" in changes:
            row.metadata_ = changes["metadata"]
        self._session.flush()
        return _to_domain(row)


def _to_domain(row: CaseInterventionRow) -> CaseIntervention:
    return CaseIntervention(
        id=row.id,
        case_id=row.case_id,
        workspace_id=row.workspace_id,
        name=row.name,
        description=row.description,
        intervention_type=row.intervention_type,
        priority=row.priority,
        status=row.status,
        rationale=row.rationale,
        expected_impact=row.expected_impact,
        timeframe=row.timeframe,
        evidence_refs=tuple(row.evidence_refs or []),
        assumptions=tuple(row.assumptions or []),
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
