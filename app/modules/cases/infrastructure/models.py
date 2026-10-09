from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ...rag.infrastructure.db.models import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TransformationCaseRow(Base):
    __tablename__ = "transformation_cases"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    problem_statement: Mapped[str] = mapped_column(sa.Text, nullable=False)
    desired_outcome: Mapped[str] = mapped_column(sa.Text, nullable=False)
    territory: Mapped[str | None] = mapped_column(sa.String(200), nullable=True)
    population: Mapped[str | None] = mapped_column(sa.String(500), nullable=True)
    time_horizon: Mapped[str | None] = mapped_column(sa.String(200), nullable=True)
    decision_authority: Mapped[str | None] = mapped_column(sa.String(300), nullable=True)
    status: Mapped[str] = mapped_column(
        sa.String(30), nullable=False, default="draft", server_default="draft"
    )
    success_criteria: Mapped[list] = mapped_column(
        sa.JSON, nullable=False, server_default="[]"
    )
    indicators: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    constraints: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    metadata_: Mapped[dict] = mapped_column(
        "metadata", sa.JSON, nullable=False, server_default="{}"
    )
    revision: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=1, server_default="1")
    review_date: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now, server_default=sa.text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now,
        onupdate=_utc_now, server_default=sa.text("NOW()")
    )

    __table_args__ = (
        sa.Index("ix_transformation_cases_workspace_updated", "workspace_id", "updated_at"),
    )


class TransformationCaseSourceRow(Base):
    __tablename__ = "transformation_case_sources"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        sa.String(64), sa.ForeignKey("transformation_cases.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    source_role: Mapped[str] = mapped_column(sa.String(80), nullable=False, default="evidence")
    linked_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now, server_default=sa.text("NOW()")
    )

    __table_args__ = (
        sa.UniqueConstraint("case_id", "document_id", name="uq_transformation_case_source"),
        sa.Index("ix_transformation_case_sources_workspace_case", "workspace_id", "case_id"),
    )
