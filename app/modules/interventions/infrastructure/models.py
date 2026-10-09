from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ...rag.infrastructure.db.models import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CaseInterventionRow(Base):
    __tablename__ = "case_interventions"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        sa.String(64), sa.ForeignKey("transformation_cases.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    description: Mapped[str] = mapped_column(sa.Text, nullable=False)
    intervention_type: Mapped[str] = mapped_column(sa.String(80), nullable=False)
    priority: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="medium")
    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="proposed")
    rationale: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    expected_impact: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    timeframe: Mapped[str] = mapped_column(sa.String(200), nullable=False, default="")
    evidence_refs: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    assumptions: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    metadata_: Mapped[dict] = mapped_column("metadata", sa.JSON, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now, server_default=sa.text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now,
        onupdate=_utc_now, server_default=sa.text("NOW()")
    )

    __table_args__ = (
        sa.UniqueConstraint("case_id", "name", name="uq_case_intervention_name"),
        sa.Index("ix_case_interventions_workspace_case", "workspace_id", "case_id"),
    )
