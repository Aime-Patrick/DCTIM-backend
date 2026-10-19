from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ..rag.infrastructure.db.models import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PlanningScenarioRow(Base):
    __tablename__ = "planning_scenarios"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    created_by: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    policy_id: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    name: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    year: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=2035)
    features: Mapped[dict] = mapped_column(sa.JSON, nullable=False, server_default='{"type":"FeatureCollection","features":[]}')
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=_utc_now, server_default=sa.text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now, server_default=sa.text("NOW()"))

    __table_args__ = (
        sa.Index("ix_planning_scenarios_workspace_updated", "workspace_id", "updated_at"),
        sa.Index("ix_planning_scenarios_workspace_policy", "workspace_id", "policy_id"),
    )
