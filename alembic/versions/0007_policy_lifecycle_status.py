"""Add lifecycle status to policy artifacts."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "policy_artifacts",
        sa.Column("status", sa.String(40), nullable=False, server_default="created"),
    )


def downgrade() -> None:
    op.drop_column("policy_artifacts", "status")
