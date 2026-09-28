"""0003_embedding_dim_2048_openrouter

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-18

Switch rag_chunks.embedding from vector(256) to vector(2048) for the free
OpenRouter model nvidia/nemotron-3-embed-1b:free (native dim 2048).

Existing vectors are cleared — re-ingest after upgrading.
"""
from __future__ import annotations

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE rag_chunks SET embedding = NULL")
    op.execute("ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector(2048)")


def downgrade() -> None:
    op.execute("UPDATE rag_chunks SET embedding = NULL")
    op.execute("ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector(256)")
