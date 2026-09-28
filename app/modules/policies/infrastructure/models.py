from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ...rag.infrastructure.db.models import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PolicyArtifact(Base):
    __tablename__ = "policy_artifacts"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    category: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    status: Mapped[str] = mapped_column(
        sa.String(40), nullable=False, default="created", server_default="created"
    )
    analysis_trace_id: Mapped[str | None] = mapped_column(sa.String(128), nullable=True)
    analysis_provider: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    analysis: Mapped[dict | None] = mapped_column(sa.JSON, nullable=True)
    revision: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        default=1,
        server_default="1",
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=sa.text("NOW()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
        server_default=sa.text("NOW()"),
    )

    __table_args__ = (
        sa.Index("ix_policy_artifacts_workspace_updated", "workspace_id", "updated_at"),
    )
