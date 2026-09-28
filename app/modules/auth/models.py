from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ..rag.infrastructure.db.models import Base


class WorkspaceUser(Base):
    __tablename__ = "workspace_users"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    email: Mapped[str] = mapped_column(sa.String(200), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    role: Mapped[str] = mapped_column(sa.String(40), nullable=False, server_default="viewer")
    permissions: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    is_active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.true())
