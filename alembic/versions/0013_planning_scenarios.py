"""Add workspace-scoped geographic planning scenarios."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "planning_scenarios",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False, server_default="2035"),
        sa.Column("features", sa.JSON(), nullable=False, server_default='{"type":"FeatureCollection","features":[]}'),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_planning_scenarios_workspace_id", "planning_scenarios", ["workspace_id"])
    op.create_index("ix_planning_scenarios_workspace_updated", "planning_scenarios", ["workspace_id", "updated_at"])


def downgrade() -> None:
    op.drop_index("ix_planning_scenarios_workspace_updated", table_name="planning_scenarios")
    op.drop_index("ix_planning_scenarios_workspace_id", table_name="planning_scenarios")
    op.drop_table("planning_scenarios")
