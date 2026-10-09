"""Add workspace-scoped transformation cases and evidence links."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision: str = "0010"
down_revision: str | None = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transformation_cases",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("problem_statement", sa.Text(), nullable=False),
        sa.Column("desired_outcome", sa.Text(), nullable=False),
        sa.Column("territory", sa.String(length=200), nullable=True),
        sa.Column("population", sa.String(length=500), nullable=True),
        sa.Column("time_horizon", sa.String(length=200), nullable=True),
        sa.Column("decision_authority", sa.String(length=300), nullable=True),
        sa.Column("status", sa.String(length=30), server_default="draft", nullable=False),
        sa.Column("success_criteria", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("indicators", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("constraints", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("metadata", sa.JSON(), server_default="{}", nullable=False),
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column("review_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_transformation_cases_workspace_id", "transformation_cases", ["workspace_id"]
    )
    op.create_index(
        "ix_transformation_cases_workspace_updated",
        "transformation_cases",
        ["workspace_id", "updated_at"],
    )

    op.create_table(
        "transformation_case_sources",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("source_role", sa.String(length=80), server_default="evidence", nullable=False),
        sa.Column("linked_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["transformation_cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("case_id", "document_id", name="uq_transformation_case_source"),
    )
    op.create_index(
        "ix_transformation_case_sources_workspace_id",
        "transformation_case_sources",
        ["workspace_id"],
    )
    op.create_index(
        "ix_transformation_case_sources_workspace_case",
        "transformation_case_sources",
        ["workspace_id", "case_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_transformation_case_sources_workspace_case", table_name="transformation_case_sources")
    op.drop_index("ix_transformation_case_sources_workspace_id", table_name="transformation_case_sources")
    op.drop_table("transformation_case_sources")
    op.drop_index("ix_transformation_cases_workspace_updated", table_name="transformation_cases")
    op.drop_index("ix_transformation_cases_workspace_id", table_name="transformation_cases")
    op.drop_table("transformation_cases")
