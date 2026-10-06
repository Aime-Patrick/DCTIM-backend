"""PgVectorStore — production VectorStore backed by PostgreSQL + pgvector.

Implements the ``VectorStore`` port defined in ``rag.ports``.  The application
service is unaware of this class; it only depends on the port protocol.

Retrieval strategy
------------------
Supports both dense vector search via the ``<=>`` pgvector operator and hybrid
retrieval fusing dense vector similarity with PostgreSQL full-text search (BM25/ts_rank_cd)
via Reciprocal Rank Fusion (RRF).

The query in ``search()`` and ``search_hybrid()`` always includes a ``workspace_id``
equality filter so every query is strictly scoped to a single tenant.
"""
from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, literal_column, select
from sqlalchemy.orm import Session

from ..domain import Chunk, RetrievedChunk
from ..ports import VectorStore
from .db.models import RagChunk
from .reranker import reciprocal_rank_fusion


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
        """Batch-upsert chunks and their embeddings."""
        if len(chunks) != len(vectors):
            raise ValueError("each chunk must have exactly one vector")

        for chunk, vector in zip(chunks, vectors):
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
        """Return the *limit* most similar chunks for *workspace_id* using dense vector cosine distance."""
        if limit <= 0:
            return []

        # Build the query vector as a Postgres literal.
        vector_literal = "[" + ",".join(str(float(v)) for v in query_vector) + "]"

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
                    score=round(1.0 - float(row.distance), 6),
                )
            )
        return results

    def search_lexical(
        self,
        workspace_id: str,
        query_text: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Full-text lexical search using PostgreSQL plainto_tsquery."""
        if limit <= 0 or not query_text.strip():
            return []

        try:
            ts_query = func.plainto_tsquery("english", query_text.strip())
            ts_vector = func.to_tsvector("english", RagChunk.content)
            rank_col = func.ts_rank_cd(ts_vector, ts_query).label("rank")

            stmt = (
                select(RagChunk, rank_col)
                .where(
                    RagChunk.workspace_id == workspace_id,
                    ts_vector.op("@@")(ts_query),
                )
                .order_by(rank_col.desc())
                .limit(limit)
            )

            rows = self._session.execute(stmt).all()
        except Exception:
            return []

        results: list[RetrievedChunk] = []
        for row in rows:
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
                    score=min(1.0, round(float(row.rank), 4)),
                )
            )
        return results

    def search_hybrid(
        self,
        workspace_id: str,
        query_text: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Hybrid search combining dense vector embeddings and full-text keyword ranking via RRF."""
        if limit <= 0:
            return []

        dense_hits = self.search(workspace_id, query_vector, limit=limit * 2)
        lexical_hits = self.search_lexical(workspace_id, query_text, limit=limit * 2)

        if not lexical_hits:
            return dense_hits[:limit]
        if not dense_hits:
            return lexical_hits[:limit]

        fused = reciprocal_rank_fusion(dense_hits, lexical_hits)
        return fused[:limit]

    def expand_neighbors(
        self,
        workspace_id: str,
        hits: Sequence[RetrievedChunk],
        window: int = 1,
    ) -> list[RetrievedChunk]:
        """Add adjacent chunks from the same document for boundary-safe context."""
        if not hits or window <= 0:
            return list(hits)

        document_ids = {hit.chunk.document_id for hit in hits}
        rows = self._session.scalars(
            select(RagChunk).where(
                RagChunk.workspace_id == workspace_id,
                RagChunk.document_id.in_(document_ids),
            )
        ).all()
        by_document: dict[str, list[RagChunk]] = {}
        for row in rows:
            by_document.setdefault(row.document_id, []).append(row)

        scores = {hit.chunk.id: hit.score for hit in hits}
        chunks = {hit.chunk.id: hit.chunk for hit in hits}
        for hit in hits:
            for row in by_document.get(hit.chunk.document_id, []):
                if abs(row.ordinal - hit.chunk.ordinal) > window:
                    continue
                scores[row.id] = max(scores.get(row.id, 0.0), round(hit.score * 0.97, 6))
                chunks[row.id] = Chunk(
                    id=row.id,
                    document_id=row.document_id,
                    workspace_id=row.workspace_id,
                    content=row.content,
                    ordinal=row.ordinal,
                    metadata=row.metadata_,
                )

        expanded = [RetrievedChunk(chunk=chunks[cid], score=score) for cid, score in scores.items()]
        expanded.sort(key=lambda item: item.score, reverse=True)
        return expanded
