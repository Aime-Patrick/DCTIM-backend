"""Persist workspace users, roles and permissions."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workspace_users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(200), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("workspace_id", sa.String(100), nullable=False),
        sa.Column("password_hash", sa.String(200), nullable=False),
        sa.Column("role", sa.String(40), nullable=False, server_default="viewer"),
        sa.Column("permissions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index("ix_workspace_users_email", "workspace_users", ["email"], unique=True)
    op.create_index("ix_workspace_users_workspace_id", "workspace_users", ["workspace_id"])


def downgrade() -> None:
    op.drop_index("ix_workspace_users_workspace_id", table_name="workspace_users")
    op.drop_index("ix_workspace_users_email", table_name="workspace_users")
    op.drop_table("workspace_users")
