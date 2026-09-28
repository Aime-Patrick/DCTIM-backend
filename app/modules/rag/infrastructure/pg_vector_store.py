"""PgVectorStore — production VectorStore backed by PostgreSQL + pgvector.

Implements the ``VectorStore`` port defined in ``rag.ports``.  The application
service is unaware of this class; it only depends on the port protocol.

Retrieval strategy
------------------
Phase 1 uses exact cosine similarity via the ``<=>`` pgvector operator and a
plain ``ORDER BY / LIMIT`` query.  This is correct for small-to-medium corpora
and avoids approximate-search tuning before we have real data.

A follow-up migration can add an HNSW index once:
  - the corpus exceeds ~50 k chunks, OR
  - p95 query latency measurements justify the switch.

The query in ``search()`` always includes a ``workspace_id`` equality filter so
the index scan is scoped to a single tenant even without row-level security.
"""
from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import literal_column, select
from sqlalchemy.orm import Session

from ..domain import Chunk, RetrievedChunk
from ..ports import VectorStore
from .db.models import RagChunk


class PgVectorStore:
    """Persistent vector store using PostgreSQL + pgvector.

    Parameters
    ----------
    session:
        A SQLAlchemy ``Session`` scoped to the current request.  The caller
        (dependency injection) owns the transaction; this class never commits.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # VectorStore port implementation
    # ------------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Batch-upsert chunks and their embeddings.

        This method is a thin write path used only by the in-process ingestion
        flow.  The full persistence path (document row + job tracking) goes
        through ``DocumentRepository.replace_chunks``; this method exists to
        satisfy the ``VectorStore`` protocol so the application service can
        call it without knowing about the repository.

        In the persistent stack, ``RagService.ingest`` delegates the actual
        insert to the repository, so this method delegates to the repository
        as well via the shared session.  If called directly (e.g. in tests
        that bypass the repository), it performs a minimal insert.
        """
        if len(chunks) != len(vectors):
            raise ValueError("each chunk must have exactly one vector")

        for chunk, vector in zip(chunks, vectors):
            # Use a PostgreSQL upsert (INSERT ... ON CONFLICT DO UPDATE) so
            # repeated calls with the same chunk id are idempotent.
            row = self._session.get(RagChunk, chunk.id)
            if row is not None:
                row.content = chunk.content
                row.ordinal = chunk.ordinal
                row.metadata_ = chunk.metadata
                row.embedding = list(vector)
            else:
                self._session.add(
                    RagChunk(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        workspace_id=chunk.workspace_id,
                        ordinal=chunk.ordinal,
                        content=chunk.content,
                        token_count=len(chunk.content.split()),
                        metadata_=chunk.metadata,
                        embedding=list(vector),
                    )
                )
        self._session.flush()

    def search(
        self,
        workspace_id: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Return the *limit* most similar chunks for *workspace_id*.

        Uses the pgvector cosine distance operator ``<=>`` so lower values
        mean more similar.  We convert distance → similarity score so the
        domain type stays consistent with the in-memory adapter.

        The ``workspace_id`` equality filter is a hard requirement and must
        never be removed; it is the primary tenant isolation mechanism at the
        database layer.
        """
        if limit <= 0:
            return []

        # Build the query vector as a Postgres literal.
        vector_literal = "[" + ",".join(str(float(v)) for v in query_vector) + "]"

        # Use literal_column + .label() so SQLAlchemy registers the alias
        # and row.distance is accessible on the result rows.
        distance_col = literal_column(
            f"embedding <=> '{vector_literal}'::vector"
        ).label("distance")

        stmt = (
            select(RagChunk, distance_col)
            .where(
                RagChunk.workspace_id == workspace_id,
                RagChunk.embedding.is_not(None),
            )
            .order_by(distance_col)
            .limit(limit)
        )

        rows = self._session.execute(stmt).all()

        results: list[RetrievedChunk] = []
        for row in rows:
            if row.distance is None:
                # Skip chunks whose embedding was cleared (e.g. after dim migration).
                continue
            results.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=row.RagChunk.id,
                        document_id=row.RagChunk.document_id,
                        workspace_id=row.RagChunk.workspace_id,
                        content=row.RagChunk.content,
                        ordinal=row.RagChunk.ordinal,
                        metadata=row.RagChunk.metadata_,
                    ),
                    # cosine distance → cosine similarity
                    score=round(1.0 - float(row.distance), 6),
                )
            )
        return results
