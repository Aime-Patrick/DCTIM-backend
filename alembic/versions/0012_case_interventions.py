"""Add approved interventions for transformation cases."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision: str = "0012"
down_revision: str | None = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "case_interventions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("intervention_type", sa.String(length=80), nullable=False),
        sa.Column("priority", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("expected_impact", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.String(length=200), nullable=False),
        sa.Column("evidence_refs", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("assumptions", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("metadata", sa.JSON(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["transformation_cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("case_id", "name", name="uq_case_intervention_name"),
    )
    op.create_index("ix_case_interventions_workspace_id", "case_interventions", ["workspace_id"])
    op.create_index(
        "ix_case_interventions_workspace_case",
        "case_interventions",
        ["workspace_id", "case_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_case_interventions_workspace_case", table_name="case_interventions")
    op.drop_index("ix_case_interventions_workspace_id", table_name="case_interventions")
    op.drop_table("case_interventions")
