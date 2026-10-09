from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..application import IndicatorCreateCommand
from ..domain import CaseIndicator
from .models import CaseIndicatorRow


class IndicatorRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_indicator(
        self, workspace_id: str, case_id: str, command: IndicatorCreateCommand
    ) -> CaseIndicator | None:
        row = CaseIndicatorRow(
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
            source_refs=list(command.source_refs),
            measurement_date=command.measurement_date,
            metadata_=dict(command.metadata or {}),
        )
        self._session.add(row)
        self._session.flush()
        return _to_domain(row)

    def list_indicators(self, workspace_id: str, case_id: str) -> list[CaseIndicator]:
        rows = self._session.scalars(
            select(CaseIndicatorRow)
            .where(
                CaseIndicatorRow.workspace_id == workspace_id,
                CaseIndicatorRow.case_id == case_id,
            )
            .order_by(CaseIndicatorRow.name.asc(), CaseIndicatorRow.id.asc())
        ).all()
        return [_to_domain(row) for row in rows]

    def get_indicator(
        self, workspace_id: str, case_id: str, indicator_id: str
    ) -> CaseIndicator | None:
        row = self._session.scalar(
            select(CaseIndicatorRow).where(
                CaseIndicatorRow.id == indicator_id,
                CaseIndicatorRow.workspace_id == workspace_id,
                CaseIndicatorRow.case_id == case_id,
            )
        )
        return _to_domain(row) if row is not None else None

    def update_indicator(
        self,
        workspace_id: str,
        case_id: str,
        indicator_id: str,
        changes: dict[str, Any],
    ) -> CaseIndicator | None:
        row = self._session.scalar(
            select(CaseIndicatorRow).where(
                CaseIndicatorRow.id == indicator_id,
                CaseIndicatorRow.workspace_id == workspace_id,
                CaseIndicatorRow.case_id == case_id,
            )
        )
        if row is None:
            return None
        for field_name in (
            "name", "definition", "unit", "direction", "baseline_value", "target_value",
            "current_value", "uncertainty", "quality_status", "source_refs", "measurement_date",
        ):
            if field_name in changes:
                setattr(row, field_name, changes[field_name])
        if "metadata" in changes:
            row.metadata_ = changes["metadata"]
        self._session.flush()
        return _to_domain(row)


def _to_domain(row: CaseIndicatorRow) -> CaseIndicator:
    return CaseIndicator(
        id=row.id,
        case_id=row.case_id,
        workspace_id=row.workspace_id,
        name=row.name,
        definition=row.definition,
        unit=row.unit,
        direction=row.direction,
        baseline_value=row.baseline_value,
        target_value=row.target_value,
        current_value=row.current_value,
        uncertainty=row.uncertainty,
        quality_status=row.quality_status,
        source_refs=tuple(row.source_refs or []),
        measurement_date=row.measurement_date,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
