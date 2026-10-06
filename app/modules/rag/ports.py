from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .domain import Chunk, RetrievedChunk


class EmbeddingProvider(Protocol):
    """Small contract for swapping local or hosted embedding models."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    """Storage contract for vector records and scoped similarity search."""

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self,
        workspace_id: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]: ...

    def search_hybrid(
        self,
        workspace_id: str,
        query_text: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]: ...

    def expand_neighbors(
        self,
        workspace_id: str,
        hits: Sequence[RetrievedChunk],
        window: int = 1,
    ) -> list[RetrievedChunk]: ...


class AnswerGenerator(Protocol):
    """Contract for a grounded LLM or a deterministic development generator."""

    def generate(self, query: str, contexts: Sequence[RetrievedChunk]) -> str: ...

    def analyze(self, query: str, contexts: Sequence[RetrievedChunk]) -> dict[str, object]: ...

    def optimize_prompt(self, prompt: str) -> str: ...
