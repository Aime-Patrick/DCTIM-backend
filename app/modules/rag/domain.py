from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    DOCUMENT = "document"
    KNOWLEDGE = "knowledge"
    QA = "qa"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Document:
    id: str
    workspace_id: str
    title: str
    content: str
    source_type: SourceType
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    workspace_id: str
    content: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    title: str
    source_type: SourceType
    chunk_count: int


@dataclass(frozen=True)
class DocumentEntry:
    """Read-model row describing one ingested source in the workspace."""

    id: str
    title: str
    source_type: SourceType
    source_id: str | None
    created_at: datetime
    file_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    preview: str | None = None


@dataclass(frozen=True)
class DataSummary:
    """Workspace-wide totals regardless of the current filter/pagination."""

    total_entries: int
    total_chunks: int
    count_by_type: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentListing:
    """Page of ingested entries plus workspace summary and total."""

    summary: DataSummary
    total: int
    entries: list[DocumentEntry] = field(default_factory=list)


@dataclass(frozen=True)
class QueryTelemetry:
    embed_ms: float
    search_ms: float
    generate_ms: float
    total_ms: float
    citation_count: int
    citation_coverage: float
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    intent: str | None = None
    search_strategy: str = "hybrid_rrf"
    rerank_ms: float = 0.0


@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: tuple[RetrievedChunk, ...]
    trace_id: str
    telemetry: QueryTelemetry | None = None
    answer_mode: str = "synthesized"


class PolicyAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyAnalysisResult:
    analysis: dict[str, Any]
    citations: tuple[RetrievedChunk, ...]
    trace_id: str
    telemetry: QueryTelemetry | None = None

    @property
    def content(self) -> dict[str, Any]:
        return self.analysis
