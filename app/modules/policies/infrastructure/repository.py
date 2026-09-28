from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..domain import PolicyArtifact, PolicySummary, utc_now
from .models import PolicyArtifact as PolicyArtifactRow


class PolicyRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_artifact(
        self,
        workspace_id: str,
        title: str,
        content: str,
        category: str,
        status: str,
        analysis_trace_id: str | None,
        analysis_provider: str | None,
        analysis: dict[str, Any] | None,
    ) -> PolicyArtifact:
        row = PolicyArtifactRow(
            id=uuid4().hex,
            workspace_id=workspace_id,
            title=title,
            content=content,
            category=category,
            status=status,
            analysis_trace_id=analysis_trace_id,
            analysis_provider=analysis_provider,
            analysis=analysis,
        )
        self._session.add(row)
        self._session.flush()
        return _to_domain(row)

    def list_artifacts(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PolicySummary], int]:
        total = (
            self._session.scalar(
                select(func.count(PolicyArtifactRow.id)).where(
                    PolicyArtifactRow.workspace_id == workspace_id
                )
            )
            or 0
        )
        rows = self._session.scalars(
            select(PolicyArtifactRow)
            .where(PolicyArtifactRow.workspace_id == workspace_id)
            .order_by(PolicyArtifactRow.updated_at.desc(), PolicyArtifactRow.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return [_to_summary(row) for row in rows], int(total)

    def get_artifact(self, workspace_id: str, policy_id: str) -> PolicyArtifact | None:
        row = self._session.scalar(
            select(PolicyArtifactRow).where(
                PolicyArtifactRow.id == policy_id,
                PolicyArtifactRow.workspace_id == workspace_id,
            )
        )
        return _to_domain(row) if row is not None else None

    def update_artifact(
        self,
        workspace_id: str,
        policy_id: str,
        changes: dict[str, Any],
    ) -> PolicyArtifact | None:
        row = self._session.scalar(
            select(PolicyArtifactRow).where(
                PolicyArtifactRow.id == policy_id,
                PolicyArtifactRow.workspace_id == workspace_id,
            )
        )
        if row is None:
            return None
        for field_name in ("title", "content", "category", "status", "analysis"):
            if field_name in changes:
                setattr(row, field_name, changes[field_name])
        row.revision += 1
        row.updated_at = utc_now()
        self._session.flush()
        return _to_domain(row)


def _to_domain(row: PolicyArtifactRow) -> PolicyArtifact:
    return PolicyArtifact(
        id=row.id,
        workspace_id=row.workspace_id,
        title=row.title,
        content=row.content,
        category=row.category,
        status=row.status,
        analysis_trace_id=row.analysis_trace_id,
        analysis_provider=row.analysis_provider,
        analysis=row.analysis,
        revision=row.revision,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_summary(row: PolicyArtifactRow) -> PolicySummary:
    return PolicySummary(
        id=row.id,
        title=row.title,
        category=row.category,
        status=row.status,
        analysis_trace_id=row.analysis_trace_id,
        analysis_provider=row.analysis_provider,
        has_analysis=row.analysis is not None,
        revision=row.revision,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
