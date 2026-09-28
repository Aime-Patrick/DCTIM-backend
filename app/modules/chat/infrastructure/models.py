"""SQLAlchemy models for the chat history module.

Tables are registered on the shared ``Base`` metadata defined by the RAG
module (``app.modules.rag.infrastructure.db.models``) so Alembic sees them.
"""
from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ...rag.infrastructure.db.models import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(sa.String(200), nullable=False, default="New conversation")
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        sa.String(64),
        sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    role: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    prompt: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", sa.JSON, nullable=False, server_default="{}"
    )
    ordinal: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now
    )