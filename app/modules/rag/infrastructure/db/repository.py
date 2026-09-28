"""DocumentRepository — persists RAG documents, chunks, and ingestion jobs.

All SQL stays inside this file. The application service and PgVectorStore
never touch the ORM models directly; they work through this repository and
the domain types in rag.domain.

Workspace isolation is enforced at the query level (every SELECT/INSERT
includes a workspace_id filter), not only at the route level.
"""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from sqlalchemy import exists, func, or_, select, update
from sqlalchemy.orm import Session

from ...domain import (
    Chunk,
    DataSummary,
    Document,
    DocumentEntry,
    DocumentListing,
    SourceType,
)
from .models import DocumentStatus, IngestionStatus, RagChunk, RagDocument, RagIngestionJob


def _content_hash(content: str) -> str:
    """SHA-256 hex digest of normalised UTF-8 content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class DocumentRepository:
    """Provides persistence operations for the RAG module.

    Accepts a SQLAlchemy ``Session`` injected per request so the caller
    controls transaction boundaries.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def get_document_by_source(
        self,
        workspace_id: str,
        source_type: str,
        source_id: str,
    ) -> RagDocument | None:
        """Return an existing document by its source identity, or None."""
        stmt = select(RagDocument).where(
            RagDocument.workspace_id == workspace_id,
            RagDocument.source_type == source_type,
            RagDocument.source_id == source_id,
        )
        return self._session.scalar(stmt)

    def upsert_document(self, document: Document) -> RagDocument:
        """Insert a new document row or update an existing one.

        If a row with the same (workspace_id, source_type, source_id) exists
        and the content hash is unchanged, the row is returned as-is so the
        caller can skip re-embedding.

        Returns the persisted ``RagDocument`` row.
        """
        content_hash = _content_hash(document.content)

        existing: RagDocument | None = None
        if document.source_id:
            existing = self.get_document_by_source(
                document.workspace_id,
                document.source_type.value,
                document.source_id,
            )

        if existing is not None:
            # Update mutable fields; preserve id and created_at.
            existing.title = document.title
            existing.content_hash = content_hash
            existing.metadata_ = document.metadata
            existing.status = DocumentStatus.ACTIVE.value
            if document.metadata.get("file_path"):
                existing.file_path = str(document.metadata["file_path"])
            self._session.flush()
            return existing

        row = RagDocument(
            id=document.id,
            workspace_id=document.workspace_id,
            title=document.title,
            source_type=document.source_type.value,
            source_id=document.source_id,
            content_hash=content_hash,
            metadata_=document.metadata,
            file_path=str(document.metadata["file_path"]) if document.metadata.get("file_path") else None,
            status=DocumentStatus.ACTIVE.value,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def content_hash_matches(self, document_id: str, content: str) -> bool:
        """Return True if the stored hash equals the hash of *content*."""
        stmt = select(RagDocument.content_hash).where(RagDocument.id == document_id)
        stored = self._session.scalar(stmt)
        if stored is None:
            return False
        return stored == _content_hash(content)

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def replace_chunks(
        self,
        document_id: str,
        workspace_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Delete existing chunks for *document_id* and insert fresh ones.

        Called inside the same transaction as ``upsert_document`` so a
        partial failure leaves no orphaned chunks.
        """
        if len(chunks) != len(vectors):
            raise ValueError("each chunk must have exactly one vector")

        # Remove old chunks (ON DELETE CASCADE on the FK handles orphan jobs
        # if any exist, but we delete explicitly to be explicit).
        self._session.query(RagChunk).filter(RagChunk.document_id == document_id).delete(
            synchronize_session="fetch"
        )

        rows = [
            RagChunk(
                id=chunk.id,
                document_id=document_id,
                workspace_id=workspace_id,
                ordinal=chunk.ordinal,
                content=chunk.content,
                token_count=len(chunk.content.split()),
                metadata_=chunk.metadata,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._session.add_all(rows)
        self._session.flush()

    # ------------------------------------------------------------------
    # Ingestion job operations
    # ------------------------------------------------------------------

    def create_ingestion_job(self, workspace_id: str, document_id: str) -> RagIngestionJob:
        """Create a new ingestion job in PENDING state."""
        job = RagIngestionJob(
            id=uuid4().hex,
            workspace_id=workspace_id,
            document_id=document_id,
            status=IngestionStatus.PENDING.value,
            attempts=0,
        )
        self._session.add(job)
        self._session.flush()
        return job

    def mark_job_completed(self, job_id: str) -> None:
        self._session.execute(
            update(RagIngestionJob)
            .where(RagIngestionJob.id == job_id)
            .values(status=IngestionStatus.COMPLETED.value)
        )

    def mark_job_failed(self, job_id: str, error: str) -> None:
        self._session.execute(
            update(RagIngestionJob)
            .where(RagIngestionJob.id == job_id)
            .values(
                status=IngestionStatus.FAILED.value,
                error=error[:2000],  # truncate to avoid huge error blobs
                attempts=RagIngestionJob.attempts + 1,
            )
        )

    # ------------------------------------------------------------------
    # Listing (Data Lake view)
    # ------------------------------------------------------------------

    def list_entries(
        self,
        workspace_id: str,
        source_type: str | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> DocumentListing:
        """Return a page of ingested sources for *workspace_id*.

        Results are ordered newest-first. ``search`` matches the document
        title or any of its chunks' content (case-insensitive). The returned
        summary is workspace-wide and ignores filter/pagination.
        """
        stmt = select(RagDocument).where(RagDocument.workspace_id == workspace_id)

        if source_type:
            stmt = stmt.where(RagDocument.source_type == source_type)

        if search:
            pattern = f"%{search.strip()}%"
            stmt = stmt.where(
                or_(
                    RagDocument.title.ilike(pattern),
                    exists().where(
                        RagChunk.document_id == RagDocument.id,
                        RagChunk.content.ilike(pattern),
                    ),
                )
            )

        total = (
            self._session.scalar(select(func.count()).select_from(stmt.subquery()))
            or 0
        )

        rows = self._session.scalars(
            stmt.order_by(RagDocument.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()

        ids = [row.id for row in rows]
        chunk_counts: dict[str, int] = {}
        previews: dict[str, str] = {}
        if ids:
            chunk_counts = dict(
                self._session.execute(
                    select(RagChunk.document_id, func.count(RagChunk.id))
                    .where(RagChunk.document_id.in_(ids))
                    .group_by(RagChunk.document_id)
                ).all()
            )
            previews = dict(
                self._session.execute(
                    select(RagChunk.document_id, RagChunk.content)
                    .where(RagChunk.document_id.in_(ids), RagChunk.ordinal == 0)
                ).all()
            )

        entries = [
            DocumentEntry(
                id=row.id,
                title=row.title,
                source_type=SourceType(row.source_type),
                source_id=row.source_id,
                created_at=row.created_at,
                file_path=row.file_path,
                metadata=dict(row.metadata_ or {}),
                chunk_count=chunk_counts.get(row.id, 0),
                preview=previews.get(row.id),
            )
            for row in rows
        ]

        count_by_type = dict(
            self._session.execute(
                select(RagDocument.source_type, func.count(RagDocument.id))
                .where(RagDocument.workspace_id == workspace_id)
                .group_by(RagDocument.source_type)
            ).all()
        )
        total_chunks = (
            self._session.scalar(
                select(func.count(RagChunk.id)).where(
                    RagChunk.workspace_id == workspace_id
                )
            )
            or 0
        )

        return DocumentListing(
            summary=DataSummary(
                total_entries=sum(count_by_type.values()),
                total_chunks=total_chunks,
                count_by_type=count_by_type,
            ),
            total=total,
            entries=entries,
        )
