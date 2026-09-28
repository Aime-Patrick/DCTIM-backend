"""SQLAlchemy ORM models for the RAG module.

All tables are prefixed with `rag_` to avoid collisions with future modules.
The `embedding` column type comes from the `pgvector` package and maps to
the PostgreSQL `vector(n)` type; dimension is set at table-creation time and
must match the configured embedding provider dimension.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class IngestionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# rag_documents
# ---------------------------------------------------------------------------

class RagDocument(Base):
    """Source-level identity for a piece of ingested knowledge.

    `content_hash` enables idempotent re-ingestion: identical content does not
    create a new version or trigger embedding work.
    """

    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    source_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    source_id: Mapped[str | None] = mapped_column(sa.String(200), nullable=True)
    content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    # Relative path under the uploads/ root, e.g. "workspace-a/abc123_report.pdf".
    # NULL for text/Q&A entries that were never stored as a file.
    file_path: Mapped[str | None] = mapped_column(sa.String(500), nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", sa.JSON, nullable=False, server_default="{}"
    )
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default=DocumentStatus.ACTIVE.value
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )

    __table_args__ = (
        # Workspace + source identity index for idempotent upsert lookups.
        sa.Index("ix_rag_documents_workspace_source", "workspace_id", "source_type", "source_id"),
    )


# ---------------------------------------------------------------------------
# rag_chunks
# ---------------------------------------------------------------------------

class RagChunk(Base):
    """A single retrieval unit produced from a parent document.

    The `embedding` column stores the dense vector. Its dimension is supplied
    at runtime from `Settings.embedding_dimension`; Alembic records the exact
    value used when the migration was generated so schema drift is caught.
    """

    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        sa.String(64),
        sa.ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    token_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", sa.JSON, nullable=False, server_default="{}"
    )
    # Dimension matches Settings.embedding_dimension (OpenRouter nemotron free = 2048).
    # Change this default ONLY via a new migration after updating the provider.
    embedding: Mapped[list] = mapped_column(Vector(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now
    )

    __table_args__ = (
        sa.UniqueConstraint("document_id", "ordinal", name="uq_rag_chunks_doc_ordinal"),
        # Workspace filter index used on every similarity search query.
        sa.Index("ix_rag_chunks_workspace_document", "workspace_id", "document_id"),
    )


# ---------------------------------------------------------------------------
# rag_ingestion_jobs
# ---------------------------------------------------------------------------

class RagIngestionJob(Base):
    """Tracks retryable ingestion operations.

    A job is created when a document is submitted and updated as the pipeline
    progresses. Failures are surfaced here rather than propagated directly to
    the caller so async ingestion can be added without changing the API shape.
    """

    __tablename__ = "rag_ingestion_jobs"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(
        sa.String(64),
        sa.ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default=IngestionStatus.PENDING.value, index=True
    )
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )
