"""RagService — application service coordinating ingestion and query.

The service depends only on ports (protocols), never on infrastructure
implementations.  Dependency injection in ``dependencies.py`` wires the
correct adapters for each environment.

When a ``DocumentRepository`` is supplied (production/staging), ingestion
persists the document row, chunks, embeddings, and an ingestion job in a
single unit of work.  When it is omitted (unit tests), the in-memory
``VectorStore`` adapter handles state directly — no database required.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import ValidationError

from .domain import (
    Chunk,
    DataSummary,
    Document,
    DocumentListing,
    IngestResult,
    PolicyAnalysisError,
    PolicyAnalysisResult,
    QueryResult,
    QueryTelemetry,
    SourceType,
)
from .evidence import INSUFFICIENT_EVIDENCE_MESSAGE, filter_by_min_score
from .ports import AnswerGenerator, EmbeddingProvider, VectorStore
from .schemas import PolicyAnalysisContent

# Avoid a hard import so the service stays importable without SQLAlchemy
# installed (unit-test environments, CI without DB dependencies).
try:
    from .infrastructure.db.repository import DocumentRepository
except ImportError:  # pragma: no cover
    DocumentRepository = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class IngestCommand:
    title: str
    content: str
    source_type: SourceType
    source_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    document_id: str | None = None


class RagService:
    """Coordinates ingestion and query while keeping providers behind ports.

    Parameters
    ----------
    embeddings:
        Embedding provider (local hash for tests, real model for production).
    vector_store:
        Storage adapter (in-memory for tests, pgvector for production).
    answer_generator:
        LLM or demo answer generator.
    chunker:
        Text splitting function.
    document_repository:
        Optional.  When supplied, documents/chunks are persisted to PostgreSQL
        and ingestion jobs are tracked.  When None, state lives in the
        vector_store only (unit tests).
    default_top_k:
        Default number of chunks to retrieve when the caller does not specify.
    """

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        chunker: Callable[[str], list[str]],
        document_repository: "DocumentRepository | None" = None,
        default_top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._answer_generator = answer_generator
        self._chunker = chunker
        self._repository = document_repository
        self._default_top_k = default_top_k
        self._min_score = min_score

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, workspace_id: str, command: IngestCommand) -> IngestResult:
        title = command.title.strip()
        content = command.content.strip()
        if not title or not content:
            raise ValueError("title and content are required")

        document = Document(
            id=command.document_id or uuid4().hex,
            workspace_id=workspace_id,
            title=title,
            content=content,
            source_type=command.source_type,
            source_id=command.source_id,
            metadata=dict(command.metadata),
        )
        text_chunks = self._chunker(document.content)
        chunks = [
            Chunk(
                id=uuid4().hex,
                document_id=document.id,
                workspace_id=document.workspace_id,
                content=text,
                ordinal=ordinal,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "source_type": document.source_type.value,
                    "source_id": document.source_id,
                },
            )
            for ordinal, text in enumerate(text_chunks)
        ]
        vectors = self._embeddings.embed([chunk.content for chunk in chunks])

        if self._repository is not None:
            # Persistent path: document row → chunks + embeddings → job record.
            doc_row = self._repository.upsert_document(document)
            job = self._repository.create_ingestion_job(workspace_id, doc_row.id)
            try:
                self._repository.replace_chunks(doc_row.id, workspace_id, chunks, vectors)
                self._repository.mark_job_completed(job.id)
            except Exception as exc:
                self._repository.mark_job_failed(job.id, str(exc))
                raise
        else:
            # In-memory path for unit tests (no database).
            self._vector_store.upsert(chunks, vectors)

        return IngestResult(
            document_id=document.id,
            title=document.title,
            source_type=document.source_type,
            chunk_count=len(chunks),
        )

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_entries(
        self,
        workspace_id: str,
        source_type: SourceType | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> DocumentListing:
        """List ingested sources for *workspace_id*, newest first.

        Without a ``DocumentRepository`` (in-memory/test mode) there is no
        persistent index to list from, so an empty listing is returned.
        """
        if self._repository is None:
            return DocumentListing(
                summary=DataSummary(total_entries=0, total_chunks=0),
                total=0,
            )
        return self._repository.list_entries(
            workspace_id,
            source_type=source_type.value if source_type else None,
            search=search,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        workspace_id: str,
        query: str,
        top_k: int | None = None,
    ) -> QueryResult:
        import time

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query is required")
        limit = top_k if top_k is not None else self._default_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        t0 = time.perf_counter()
        query_vector = self._embeddings.embed([normalized_query])[0]
        t1 = time.perf_counter()
        contexts = self._vector_store.search(workspace_id, query_vector, limit)
        contexts = filter_by_min_score(contexts, self._min_score)
        t2 = time.perf_counter()

        def _estimate_tokens(text: str) -> int:
            # Rough ~4 chars/token heuristic for cost metering without a tokenizer.
            return max(1, len(text) // 4) if text else 0

        if not contexts:
            total_ms = (t2 - t0) * 1000
            prompt_tokens = _estimate_tokens(normalized_query)
            completion_tokens = _estimate_tokens(INSUFFICIENT_EVIDENCE_MESSAGE)
            return QueryResult(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=(),
                trace_id=uuid4().hex,
                telemetry=QueryTelemetry(
                    embed_ms=round((t1 - t0) * 1000, 2),
                    search_ms=round((t2 - t1) * 1000, 2),
                    generate_ms=0.0,
                    total_ms=round(total_ms, 2),
                    citation_count=0,
                    citation_coverage=0.0,
                    estimated_prompt_tokens=prompt_tokens,
                    estimated_completion_tokens=completion_tokens,
                ),
            )

        answer = self._answer_generator.generate(normalized_query, contexts)
        t3 = time.perf_counter()
        citation_count = len(contexts)
        evidence_chars = sum(len(item.chunk.content) for item in contexts)
        prompt_tokens = _estimate_tokens(normalized_query) + max(1, evidence_chars // 4)
        completion_tokens = _estimate_tokens(answer)
        return QueryResult(
            answer=answer,
            citations=tuple(contexts),
            trace_id=uuid4().hex,
            telemetry=QueryTelemetry(
                embed_ms=round((t1 - t0) * 1000, 2),
                search_ms=round((t2 - t1) * 1000, 2),
                generate_ms=round((t3 - t2) * 1000, 2),
                total_ms=round((t3 - t0) * 1000, 2),
                citation_count=citation_count,
                citation_coverage=1.0 if citation_count else 0.0,
                estimated_prompt_tokens=prompt_tokens,
                estimated_completion_tokens=completion_tokens,
            ),
        )

    def analyze(
        self,
        workspace_id: str,
        query: str,
        top_k: int | None = None,
    ) -> PolicyAnalysisResult:
        import time

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query is required")
        limit = top_k if top_k is not None else self._default_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        t0 = time.perf_counter()
        query_vector = self._embeddings.embed([normalized_query])[0]
        t1 = time.perf_counter()
        contexts = self._vector_store.search(workspace_id, query_vector, limit)
        contexts = filter_by_min_score(contexts, self._min_score)
        t2 = time.perf_counter()

        def _estimate_tokens(text: str) -> int:
            return max(1, len(text) // 4) if text else 0

        if not contexts:
            content = PolicyAnalysisContent(
                policy_name=normalized_query[:200],
                category="general",
                summary=INSUFFICIENT_EVIDENCE_MESSAGE,
                evidence_status="insufficient",
                confidence=0.0,
                analysis_basis="insufficient_evidence",
                feasibility={
                    "score": None,
                    "rationale": "No relevant evidence was retrieved for this workspace.",
                },
                likelihood={
                    "score": None,
                    "rationale": "No relevant evidence was retrieved for this workspace.",
                },
            )
            total_ms = (t2 - t0) * 1000
            return PolicyAnalysisResult(
                analysis=content.model_dump(mode="json"),
                citations=(),
                trace_id=uuid4().hex,
                telemetry=QueryTelemetry(
                    embed_ms=round((t1 - t0) * 1000, 2),
                    search_ms=round((t2 - t1) * 1000, 2),
                    generate_ms=0.0,
                    total_ms=round(total_ms, 2),
                    citation_count=0,
                    citation_coverage=0.0,
                    estimated_prompt_tokens=_estimate_tokens(normalized_query),
                    estimated_completion_tokens=_estimate_tokens(INSUFFICIENT_EVIDENCE_MESSAGE),
                ),
            )

        raw_analysis = self._answer_generator.analyze(normalized_query, contexts)
        try:
            content = PolicyAnalysisContent.model_validate(raw_analysis)
        except (ValidationError, TypeError, ValueError) as exc:
            raise PolicyAnalysisError("policy analysis provider returned invalid data") from exc
        analysis = content.model_dump(mode="json")
        t3 = time.perf_counter()
        citation_count = len(contexts)
        evidence_chars = sum(len(item.chunk.content) for item in contexts)
        prompt_tokens = _estimate_tokens(normalized_query) + max(1, evidence_chars // 4)
        completion_tokens = _estimate_tokens(json.dumps(analysis, ensure_ascii=False))
        return PolicyAnalysisResult(
            analysis=analysis,
            citations=tuple(contexts),
            trace_id=uuid4().hex,
            telemetry=QueryTelemetry(
                embed_ms=round((t1 - t0) * 1000, 2),
                search_ms=round((t2 - t1) * 1000, 2),
                generate_ms=round((t3 - t2) * 1000, 2),
                total_ms=round((t3 - t0) * 1000, 2),
                citation_count=citation_count,
                citation_coverage=1.0 if citation_count else 0.0,
                estimated_prompt_tokens=prompt_tokens,
                estimated_completion_tokens=completion_tokens,
            ),
        )
