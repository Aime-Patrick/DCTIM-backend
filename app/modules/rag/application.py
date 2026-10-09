"""RagService — application service coordinating ingestion, hybrid retrieval, re-ranking, and query.

The service depends only on ports (protocols), never on concrete infrastructure
implementations. Dependency injection wires the correct adapters for each environment.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
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
from .evidence import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    GroundingValidationError,
    filter_by_min_score,
    validate_analysis_references,
)
from .infrastructure.reranker import HybridRRFReRanker, Reranker
from .text import source_locations
from .intent import classify_intent
from .ports import AnswerGenerator, EmbeddingProvider, VectorStore
from .schemas import PolicyAnalysisContent

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
    """Coordinates ingestion, hybrid search, intent classification, re-ranking, and generation.

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
        Optional repository for persistence.
    reranker:
        Optional re-ranker for hybrid rank fusion & diversity tuning.
    default_top_k:
        Default number of chunks to retrieve when unspecified.
    min_score:
        Minimum score threshold for retrieved chunks.
    """

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        chunker: Callable[[str], list[str]],
        document_repository: "DocumentRepository | None" = None,
        reranker: Reranker | None = None,
        default_top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._answer_generator = answer_generator
        self._chunker = chunker
        self._repository = document_repository
        self._reranker = reranker or HybridRRFReRanker()
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
                    "source_locations": source_locations(text),
                },
            )
            for ordinal, text in enumerate(text_chunks)
        ]
        vectors = self._embeddings.embed([chunk.content for chunk in chunks])

        if self._repository is not None:
            doc_row = self._repository.upsert_document(document)
            job = self._repository.create_ingestion_job(workspace_id, doc_row.id)
            try:
                self._repository.replace_chunks(doc_row.id, workspace_id, chunks, vectors)
                self._repository.mark_job_completed(job.id)
            except Exception as exc:
                self._repository.mark_job_failed(job.id, str(exc))
                raise
        else:
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
    # Query (Hybrid Search + Intent Classification + Re-Ranking)
    # ------------------------------------------------------------------

    def query(
        self,
        workspace_id: str,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> QueryResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query is required")
        limit = top_k if top_k is not None else self._default_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        intent_result = classify_intent(normalized_query)

        t0 = time.perf_counter()
        query_vector = self._embeddings.embed([normalized_query])[0]
        t1 = time.perf_counter()

        search_query_text = intent_result.expanded_query or normalized_query
        if hasattr(self._vector_store, "search_hybrid"):
            raw_contexts = self._vector_store.search_hybrid(
                workspace_id, search_query_text, query_vector, limit * 2, document_ids=document_ids
            )
        else:
            raw_contexts = self._vector_store.search(
                workspace_id, query_vector, limit * 2, document_ids=document_ids
            )
        expand_neighbors = getattr(self._vector_store, "expand_neighbors", None)
        if expand_neighbors is not None:
            raw_contexts = expand_neighbors(workspace_id, raw_contexts, window=1)
        t2 = time.perf_counter()

        contexts = self._reranker.rerank(normalized_query, raw_contexts, limit)
        t_rerank = time.perf_counter()
        contexts = filter_by_min_score(contexts, self._min_score)

        def _estimate_tokens(text: str) -> int:
            return max(1, len(text) // 4) if text else 0

        if not contexts:
            total_ms = (t_rerank - t0) * 1000
            prompt_tokens = _estimate_tokens(normalized_query)
            completion_tokens = _estimate_tokens(INSUFFICIENT_EVIDENCE_MESSAGE)
            return QueryResult(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=(),
                trace_id=uuid4().hex,
                answer_mode="insufficient",
                telemetry=QueryTelemetry(
                    embed_ms=round((t1 - t0) * 1000, 2),
                    search_ms=round((t2 - t1) * 1000, 2),
                    generate_ms=0.0,
                    total_ms=round(total_ms, 2),
                    citation_count=0,
                    citation_coverage=0.0,
                    estimated_prompt_tokens=prompt_tokens,
                    estimated_completion_tokens=completion_tokens,
                    intent=intent_result.intent.value,
                    search_strategy="hybrid_rrf",
                    rerank_ms=round((t_rerank - t2) * 1000, 2),
                ),
            )

        answer = self._answer_generator.generate(normalized_query, contexts)
        answer_mode = getattr(
            self._answer_generator,
            "last_answer_mode",
            getattr(self._answer_generator, "answer_mode", "synthesized"),
        )
        t3 = time.perf_counter()
        citation_count = len(contexts)
        evidence_chars = sum(len(item.chunk.content) for item in contexts)
        prompt_tokens = _estimate_tokens(normalized_query) + max(1, evidence_chars // 4)
        completion_tokens = _estimate_tokens(answer)

        return QueryResult(
            answer=answer,
            citations=tuple(contexts),
            trace_id=uuid4().hex,
            answer_mode=answer_mode,
            telemetry=QueryTelemetry(
                embed_ms=round((t1 - t0) * 1000, 2),
                search_ms=round((t2 - t1) * 1000, 2),
                generate_ms=round((t3 - t_rerank) * 1000, 2),
                total_ms=round((t3 - t0) * 1000, 2),
                citation_count=citation_count,
                citation_coverage=1.0 if citation_count else 0.0,
                estimated_prompt_tokens=prompt_tokens,
                estimated_completion_tokens=completion_tokens,
                intent=intent_result.intent.value,
                search_strategy="hybrid_rrf",
                rerank_ms=round((t_rerank - t2) * 1000, 2),
            ),
        )

    def analyze(
        self,
        workspace_id: str,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> PolicyAnalysisResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query is required")
        limit = top_k if top_k is not None else self._default_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        intent_result = classify_intent(normalized_query)

        t0 = time.perf_counter()
        query_vector = self._embeddings.embed([normalized_query])[0]
        t1 = time.perf_counter()

        search_query_text = intent_result.expanded_query or normalized_query
        if hasattr(self._vector_store, "search_hybrid"):
            raw_contexts = self._vector_store.search_hybrid(
                workspace_id, search_query_text, query_vector, limit * 2, document_ids=document_ids
            )
        else:
            raw_contexts = self._vector_store.search(
                workspace_id, query_vector, limit * 2, document_ids=document_ids
            )
        expand_neighbors = getattr(self._vector_store, "expand_neighbors", None)
        if expand_neighbors is not None:
            raw_contexts = expand_neighbors(workspace_id, raw_contexts, window=1)
        t2 = time.perf_counter()

        contexts = self._reranker.rerank(normalized_query, raw_contexts, limit)
        t_rerank = time.perf_counter()
        contexts = filter_by_min_score(contexts, self._min_score)

        def _estimate_tokens(text: str) -> int:
            return max(1, len(text) // 4) if text else 0

        if not contexts:
            content = PolicyAnalysisContent(
                policy_name=normalized_query[:200],
                category=intent_result.category or "general",
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
            total_ms = (t_rerank - t0) * 1000
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
                    intent=intent_result.intent.value,
                    search_strategy="hybrid_rrf",
                    rerank_ms=round((t_rerank - t2) * 1000, 2),
                ),
            )

        raw_analysis = self._answer_generator.analyze(normalized_query, contexts)
        try:
            validate_analysis_references(
                raw_analysis,
                {item.chunk.id for item in contexts},
            )
            content = PolicyAnalysisContent.model_validate(raw_analysis)
        except (GroundingValidationError, ValidationError, TypeError, ValueError) as exc:
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
                generate_ms=round((t3 - t_rerank) * 1000, 2),
                total_ms=round((t3 - t0) * 1000, 2),
                citation_count=citation_count,
                citation_coverage=1.0 if citation_count else 0.0,
                estimated_prompt_tokens=prompt_tokens,
                estimated_completion_tokens=completion_tokens,
                intent=intent_result.intent.value,
                search_strategy="hybrid_rrf",
                rerank_ms=round((t_rerank - t2) * 1000, 2),
            ),
        )
