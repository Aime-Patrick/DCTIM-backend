from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

from ..domain import Chunk, RetrievedChunk


@dataclass(frozen=True)
class _VectorRecord:
    chunk: Chunk
    vector: tuple[float, ...]


class InMemoryVectorStore:
    """Simple vector store for local development.

    The workspace check happens inside the store, not only at the route, so every caller gets the
    same isolation guarantee. Replace this adapter with pgvector/Qdrant/etc. without changing the
    RAG use case.
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

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("query and stored vectors must have the same dimension")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

