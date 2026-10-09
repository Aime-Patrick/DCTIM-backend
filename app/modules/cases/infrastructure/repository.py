from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..application import CaseCreateCommand
from ..domain import CaseSource, TransformationCase, TransformationCaseSummary, utc_now
from .models import TransformationCaseRow, TransformationCaseSourceRow


class TransformationCaseRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_case(self, workspace_id: str, command: CaseCreateCommand) -> TransformationCase:
        row = TransformationCaseRow(
            id=uuid4().hex,
            workspace_id=workspace_id,
            title=command.title,
            problem_statement=command.problem_statement,
            desired_outcome=command.desired_outcome,
            territory=command.territory,
            population=command.population,
            time_horizon=command.time_horizon,
            decision_authority=command.decision_authority,
            status=command.status,
            success_criteria=list(command.success_criteria),
            indicators=list(command.indicators),
            constraints=list(command.constraints),
            metadata_=dict(command.metadata or {}),
            review_date=command.review_date,
        )
        self._session.add(row)
        self._session.flush()
        return _to_domain(row, ())

    def list_cases(
        self, workspace_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[list[TransformationCaseSummary], int]:
        total = self._session.scalar(
            select(func.count(TransformationCaseRow.id)).where(
                TransformationCaseRow.workspace_id == workspace_id
            )
        ) or 0
        rows = self._session.scalars(
            select(TransformationCaseRow)
            .where(TransformationCaseRow.workspace_id == workspace_id)
            .order_by(TransformationCaseRow.updated_at.desc(), TransformationCaseRow.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        summaries = [
            _to_summary(row, self._source_count(workspace_id, row.id)) for row in rows
        ]
        return summaries, int(total)

    def get_case(self, workspace_id: str, case_id: str) -> TransformationCase | None:
        row = self._session.scalar(
            select(TransformationCaseRow).where(
                TransformationCaseRow.id == case_id,
                TransformationCaseRow.workspace_id == workspace_id,
            )
        )
        if row is None:
            return None
        sources = self._session.scalars(
            select(TransformationCaseSourceRow)
            .where(
                TransformationCaseSourceRow.case_id == case_id,
                TransformationCaseSourceRow.workspace_id == workspace_id,
            )
            .order_by(TransformationCaseSourceRow.linked_at.asc())
        ).all()
        return _to_domain(row, tuple(_source_to_domain(source) for source in sources))

    def update_case(
        self, workspace_id: str, case_id: str, changes: dict[str, Any]
    ) -> TransformationCase | None:
        row = self._session.scalar(
            select(TransformationCaseRow).where(
                TransformationCaseRow.id == case_id,
                TransformationCaseRow.workspace_id == workspace_id,
            )
        )
        if row is None:
            return None
        for field_name in (
            "title", "problem_statement", "desired_outcome", "territory", "population",
            "time_horizon", "decision_authority", "status", "success_criteria",
            "indicators", "constraints", "review_date",
        ):
            if field_name in changes:
                setattr(row, field_name, changes[field_name])
        if "metadata" in changes:
            row.metadata_ = changes["metadata"]
        row.revision += 1
        row.updated_at = utc_now()
        self._session.flush()
        return self.get_case(workspace_id, case_id)

    def link_source(
        self, workspace_id: str, case_id: str, document_id: str, source_role: str
    ) -> CaseSource | None:
        case_exists = self._session.scalar(
            select(TransformationCaseRow.id).where(
                TransformationCaseRow.id == case_id,
                TransformationCaseRow.workspace_id == workspace_id,
            )
        )
        if case_exists is None:
            return None
        existing = self._session.scalar(
            select(TransformationCaseSourceRow).where(
                TransformationCaseSourceRow.case_id == case_id,
                TransformationCaseSourceRow.document_id == document_id,
                TransformationCaseSourceRow.workspace_id == workspace_id,
            )
        )
        if existing is not None:
            raise ValueError("document is already linked to this case")
        row = TransformationCaseSourceRow(
            id=uuid4().hex,
            case_id=case_id,
            workspace_id=workspace_id,
            document_id=document_id,
            source_role=source_role,
        )
        self._session.add(row)
        self._session.flush()
        return _source_to_domain(row)

    def _source_count(self, workspace_id: str, case_id: str) -> int:
        return int(
            self._session.scalar(
                select(func.count(TransformationCaseSourceRow.id)).where(
                    TransformationCaseSourceRow.case_id == case_id,
                    TransformationCaseSourceRow.workspace_id == workspace_id,
                )
            )
            or 0
        )


def _to_domain(row: TransformationCaseRow, sources: tuple[CaseSource, ...]) -> TransformationCase:
    return TransformationCase(
        id=row.id,
        workspace_id=row.workspace_id,
        title=row.title,
        problem_statement=row.problem_statement,
        desired_outcome=row.desired_outcome,
        territory=row.territory,
        population=row.population,
        time_horizon=row.time_horizon,
        decision_authority=row.decision_authority,
        status=row.status,
        success_criteria=tuple(row.success_criteria or []),
        indicators=tuple(row.indicators or []),
        constraints=tuple(row.constraints or []),
        metadata=dict(row.metadata_ or {}),
        revision=row.revision,
        review_date=row.review_date,
        created_at=row.created_at,
        updated_at=row.updated_at,
        sources=sources,
    )


def _to_summary(row: TransformationCaseRow, source_count: int) -> TransformationCaseSummary:
    return TransformationCaseSummary(
        id=row.id,
        title=row.title,
        territory=row.territory,
        status=row.status,
        revision=row.revision,
        source_count=source_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _source_to_domain(row: TransformationCaseSourceRow) -> CaseSource:
    return CaseSource(
        id=row.id,
        case_id=row.case_id,
        document_id=row.document_id,
        source_role=row.source_role,
        linked_at=row.linked_at,
    )
