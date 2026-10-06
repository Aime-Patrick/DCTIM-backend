from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..domain import Chunk, RetrievedChunk
from .reranker import reciprocal_rank_fusion


@dataclass(frozen=True)
class _VectorRecord:
    chunk: Chunk
    vector: tuple[float, ...]


class InMemoryVectorStore:
    """Simple in-memory vector and hybrid store for development and tests.

    The workspace check happens inside the store, not only at the route, so every caller gets the
    same isolation guarantee.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], _VectorRecord] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("each chunk must have exactly one vector")
        for chunk, vector in zip(chunks, vectors):
            self._records[(chunk.workspace_id, chunk.id)] = _VectorRecord(
                chunk=chunk,
                vector=tuple(vector),
            )

    def search(
        self,
        workspace_id: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit <= 0:
            return []
        matches = [
            RetrievedChunk(chunk=record.chunk, score=self._cosine(query_vector, record.vector))
            for (record_workspace, _), record in self._records.items()
            if record_workspace == workspace_id
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def search_lexical(
        self,
        workspace_id: str,
        query_text: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit <= 0 or not query_text.strip():
            return []
        tokens = set(re.findall(r"\w+", query_text.lower()))
        if not tokens:
            return []

        scored: list[RetrievedChunk] = []
        for (rec_ws, _), record in self._records.items():
            if rec_ws != workspace_id:
                continue
            content = record.chunk.content.lower()
            title = str(record.chunk.metadata.get("title", "")).lower()
            combined = f"{title} {content}"
            doc_tokens = set(re.findall(r"\w+", combined))

            overlap = len(tokens & doc_tokens)
            if overlap > 0:
                # Basic BM25-style term frequency score
                score = min(1.0, round(overlap / len(tokens), 4))
                scored.append(RetrievedChunk(chunk=record.chunk, score=score))

        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:limit]

    def search_hybrid(
        self,
        workspace_id: str,
        query_text: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
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
        """Add adjacent chunks from the same document without crossing tenants."""
        if not hits or window <= 0:
            return list(hits)

        records = [
            record
            for (record_workspace, _), record in self._records.items()
            if record_workspace == workspace_id
        ]
        by_document: dict[str, list[_VectorRecord]] = {}
        for record in records:
            by_document.setdefault(record.chunk.document_id, []).append(record)
        for document_records in by_document.values():
            document_records.sort(key=lambda item: item.chunk.ordinal)

        scores = {hit.chunk.id: hit.score for hit in hits}
        chunks = {hit.chunk.id: hit.chunk for hit in hits}
        for hit in hits:
            for record in by_document.get(hit.chunk.document_id, []):
                if abs(record.chunk.ordinal - hit.chunk.ordinal) > window:
                    continue
                scores[record.chunk.id] = max(
                    scores.get(record.chunk.id, 0.0),
                    round(hit.score * 0.97, 6),
                )
                chunks[record.chunk.id] = record.chunk

        expanded = [RetrievedChunk(chunk=chunks[cid], score=score) for cid, score in scores.items()]
        expanded.sort(key=lambda item: item.score, reverse=True)
        return expanded

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("query and stored vectors must have the same dimension")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
