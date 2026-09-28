"""Add file_path column to rag_documents

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-14 10:00:00.000000

Adds a nullable ``file_path`` column to ``rag_documents`` to store the
relative path of the uploaded file under the ``uploads/`` directory.
NULL for text/Q&A entries that were never written to disk.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_documents",
        sa.Column("file_path", sa.String(500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_documents", "file_path")
