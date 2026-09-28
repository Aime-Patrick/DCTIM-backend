"""Chat history schema: chat_conversations, chat_messages

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-21 10:00:00.000000

Notes
-----
- Conversations and messages are owned by a single user (``user_id`` = the
  verified JWT subject) and tagged with ``workspace_id`` for cross-module
  consistency. All queries filter on ``user_id``.
- Messages carry an ``ordinal`` for stable ordering within a conversation;
  ``metadata`` (JSON) stores render extras such as citations and the
  original (pre-optimization) input.
- Deleting a conversation cascades to its messages.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # chat_conversations
    # ------------------------------------------------------------------
    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("workspace_id", sa.String(100), nullable=False),
        sa.Column(
            "title",
            sa.String(200),
            nullable=False,
            server_default="New conversation",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_chat_conversations_user_id",
        "chat_conversations",
        ["user_id"],
    )
    op.create_index(
        "ix_chat_conversations_workspace_id",
        "chat_conversations",
        ["workspace_id"],
    )

    # ------------------------------------------------------------------
    # chat_messages
    # ------------------------------------------------------------------
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("trace_id", sa.String(100), nullable=True),
        sa.Column("metadata", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("ordinal", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_chat_messages_conversation_id",
        "chat_messages",
        ["conversation_id"],
    )
    op.create_index(
        "ix_chat_messages_user_id",
        "chat_messages",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_conversations")