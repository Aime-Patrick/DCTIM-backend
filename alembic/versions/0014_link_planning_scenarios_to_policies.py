"""Link Planning Studio scenarios to policy artifacts."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("planning_scenarios", sa.Column("policy_id", sa.String(length=64), nullable=True))
    op.create_index(
        "ix_planning_scenarios_workspace_policy",
        "planning_scenarios",
        ["workspace_id", "policy_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_planning_scenarios_workspace_policy", table_name="planning_scenarios")
    op.drop_column("planning_scenarios", "policy_id")
