from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ...rag.infrastructure.db.models import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CaseIndicatorRow(Base):
    __tablename__ = "case_indicators"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        sa.String(64), sa.ForeignKey("transformation_cases.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(sa.String(160), nullable=False)
    definition: Mapped[str] = mapped_column(sa.Text, nullable=False)
    unit: Mapped[str] = mapped_column(sa.String(80), nullable=False)
    direction: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="neutral")
    baseline_value: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    target_value: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    current_value: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    uncertainty: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    quality_status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default="unassessed", server_default="unassessed"
    )
    source_refs: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    measurement_date: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata", sa.JSON, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now, server_default=sa.text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now,
        onupdate=_utc_now, server_default=sa.text("NOW()")
    )

    __table_args__ = (
        sa.UniqueConstraint("case_id", "name", name="uq_case_indicator_name"),
        sa.Index("ix_case_indicators_workspace_case", "workspace_id", "case_id"),
    )
