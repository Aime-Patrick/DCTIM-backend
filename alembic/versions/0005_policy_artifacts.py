"""Add policy artifact storage.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "policy_artifacts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("workspace_id", sa.String(100), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("analysis_trace_id", sa.String(128), nullable=True),
        sa.Column("analysis_provider", sa.String(100), nullable=True),
        sa.Column("revision", sa.Integer, nullable=False, server_default="1"),
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
        "ix_policy_artifacts_workspace_id",
        "policy_artifacts",
        ["workspace_id"],
    )
    op.create_index(
        "ix_policy_artifacts_workspace_updated",
        "policy_artifacts",
        ["workspace_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_table("policy_artifacts")
