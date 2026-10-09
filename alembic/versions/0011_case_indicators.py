"""Add structured indicators for transformation cases."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision: str = "0011"
down_revision: str | None = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "case_indicators",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column("unit", sa.String(length=80), nullable=False),
        sa.Column("direction", sa.String(length=20), nullable=False),
        sa.Column("baseline_value", sa.Float(), nullable=True),
        sa.Column("target_value", sa.Float(), nullable=True),
        sa.Column("current_value", sa.Float(), nullable=True),
        sa.Column("uncertainty", sa.Float(), nullable=True),
        sa.Column("quality_status", sa.String(length=20), server_default="unassessed", nullable=False),
        sa.Column("source_refs", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("measurement_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["transformation_cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("case_id", "name", name="uq_case_indicator_name"),
    )
    op.create_index(
        "ix_case_indicators_workspace_id", "case_indicators", ["workspace_id"]
    )
    op.create_index(
        "ix_case_indicators_workspace_case", "case_indicators", ["workspace_id", "case_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_case_indicators_workspace_case", table_name="case_indicators")
    op.drop_index("ix_case_indicators_workspace_id", table_name="case_indicators")
    op.drop_table("case_indicators")
