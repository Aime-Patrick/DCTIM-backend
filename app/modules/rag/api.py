from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from urllib.parse import urlparse

from ...core.config import Settings
from ...core.identity import get_workspace_id
from ...dependencies import (
    build_answer_generator,
    get_rag_service,
    get_settings,
    require_any_permission,
    require_permission,
)
from .application import IngestCommand, RagService
from .domain import DocumentEntry, PolicyAnalysisError, RetrievedChunk, SourceType, utc_now
from .infrastructure.extractors import ExtractionError, extract_text
from .infrastructure.openai_embeddings import EmbeddingProviderError
from .infrastructure.openai_generator import AnswerGeneratorError
from .infrastructure.storage import BlobStore, StorageError, build_storage
from .schemas import (
    BatchIngestRequest,
    BatchIngestResponse,
    CitationResponse,
    DataEntry,
    DataListResponse,
    DataSummaryResponse,
    IngestRequest,
    IngestResponse,
    PromptOptimizeRequest,
    PromptOptimizeResponse,
    PolicyAnalysisRequest,
    PolicyAnalysisResponse,
    QueryRequest,
    QueryResponse,
    ScrapePreviewRequest,
    ScrapePreviewResponse,
    UploadResponse,
)
from .scraper import ScrapeError, scrape_url
from .upload_validation import (
    UploadValidationError,
    validate_extracted_text,
    validate_upload_bytes,
)

router = APIRouter(prefix="/rag", tags=["rag"])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rollback_upload(storage: BlobStore, reference: str) -> None:
    """Best-effort cleanup so a failed ingest never orphans a stored file."""
    try:
        storage.delete(reference)
    except StorageError:
        logger.exception("Could not remove stored object %s after a failed ingest", reference)


def _embedding_http_error(exc: EmbeddingProviderError) -> HTTPException:
    """Map an embedding failure onto an honest status code.

    Embedding happens before generation, so a provider outage or an exhausted free-tier
    quota must not be reported as a bad gateway: the answer model was never reached.
    """
    rate_limited = exc.status_code in (429, 402)
    return HTTPException(
        status_code=(
            status.HTTP_429_TOO_MANY_REQUESTS
            if rate_limited
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        detail=str(exc),
    )


def _get_storage(settings: Settings) -> BlobStore:
    """Build the blob store configured for this environment."""
    try:
        return build_storage(settings)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def _suggest_scrape_title(url: str, window_title: str | None) -> str:
    """A short-ish title for a scraped source: page <title> or URL-derived name."""
    candidate = (window_title or "").strip()
    if not candidate:
        path = Path(urlparse(url).path)
        stem = path.stem or path.name or urlparse(url).netloc
        candidate = stem.replace("_", " ").replace("-", " ").strip().title() or urlparse(url).netloc
    return candidate[:200]


def _entry_to_schema(entry: DocumentEntry) -> DataEntry:
    meta = entry.metadata or {}
    return DataEntry(
        id=entry.id,
        title=entry.title,
        source_type=entry.source_type,
        source_id=entry.source_id,
        created_at=entry.created_at,
        file_path=entry.file_path,
        metadata=meta,
        chunk_count=entry.chunk_count,
        preview=entry.preview,
        tags=list(meta.get("tags") or []),
        category=meta.get("category"),
    )


def _citation_to_schema(match: RetrievedChunk) -> CitationResponse:
    return CitationResponse(
        chunk_id=match.chunk.id,
        document_id=match.chunk.document_id,
        title=str(match.chunk.metadata.get("title", "Untitled source")),
        source_type=match.chunk.metadata.get("source_type", "document"),
        text=match.chunk.content,
        score=round(match.score, 6),
        metadata=match.chunk.metadata,
    )


def _provider_models(settings: Settings) -> tuple[str, str]:
    if settings.embedding_provider == "openrouter":
        embed_model = settings.openrouter_embedding_model
    elif settings.embedding_provider == "openai":
        embed_model = settings.openai_embedding_model
    else:
        embed_model = "hash"

    if settings.answer_provider == "openrouter":
        answer_model = settings.openrouter_chat_model
    elif settings.answer_provider == "openai":
        answer_model = settings.openai_chat_model
    else:
        answer_model = "demo"
    return embed_model, answer_model


# ---------------------------------------------------------------------------
# GET /api/v1/rag/entries  — list what has been ingested into this workspace
# ---------------------------------------------------------------------------

@router.get(
    "/entries",
    response_model=DataListResponse,
    summary="List ingested sources (documents, knowledge, Q&A) in the workspace.",
    dependencies=[Depends(require_any_permission("data:view", "monitoring:view", "prompt:use", "optimize:run"))],
)
def list_entries(
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[RagService, Depends(get_rag_service)],
    source_type: Annotated[SourceType | None, Query(description="Filter by source type")] = None,
    search: Annotated[str | None, Query(max_length=200, description="Search titles and content")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DataListResponse:
    listing = service.list_entries(
        workspace_id,
        source_type=source_type,
        search=search or None,
        limit=limit,
        offset=offset,
    )
    return DataListResponse(
        summary=DataSummaryResponse(
            total_entries=listing.summary.total_entries,
            total_chunks=listing.summary.total_chunks,
            count_by_type=listing.summary.count_by_type,
        ),
        total=listing.total,
        entries=[_entry_to_schema(entry) for entry in listing.entries],
    )


# ---------------------------------------------------------------------------
# POST /api/v1/rag/upload  — multipart file upload + ingest
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file, extract its text, and ingest it into the vector store.",
    dependencies=[Depends(require_permission("data:manage"))],
)
async def upload_file(
    file: Annotated[UploadFile, File(description="File to upload (.pdf .txt .md .csv .docx .json .jsonl)")],
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[RagService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadResponse:
    filename = file.filename or "upload"
    data = await file.read()
    try:
        filename = validate_upload_bytes(
            filename, data, max_bytes=settings.max_upload_bytes
        )
    except UploadValidationError as exc:
        detail = str(exc)
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "exceeds" in detail.lower() else status.HTTP_422_UNPROCESSABLE_ENTITY
        if "Unsupported file type" in detail:
            code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        raise HTTPException(status_code=code, detail=detail) from exc

    # Extract text
    try:
        content = extract_text(filename, data)
        content = validate_extracted_text(content, max_chars=settings.max_extracted_chars)
    except ExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Persist the raw file first so the storage reference can travel with the
    # document metadata — that is what ends up in rag_documents.file_path.
    document_id = uuid4().hex
    storage = _get_storage(settings)
    try:
        stored = storage.save(workspace_id, document_id, filename, data)
    except StorageError as exc:
        logger.exception("Upload storage failed for workspace %s", workspace_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Document storage failed: {exc}",
        ) from exc

    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    try:
        result = service.ingest(
            workspace_id,
            IngestCommand(
                title=title,
                content=content,
                source_type=SourceType.DOCUMENT,
                source_id=filename,
                metadata={
                    "original_filename": filename,
                    "file_path": stored.reference,
                    "storage_provider": stored.provider,
                },
                document_id=document_id,
            ),
        )
    except ValueError as exc:
        _rollback_upload(storage, stored.reference)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EmbeddingProviderError as exc:
        _rollback_upload(storage, stored.reference)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception:
        _rollback_upload(storage, stored.reference)
        raise

    return UploadResponse(
        document_id=result.document_id,
        title=result.title,
        filename=filename,
        file_path=stored.reference,
        file_url=stored.url,
        source_type=result.source_type,
        chunk_count=result.chunk_count,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/rag/ingest  — plain text / JSON body ingest
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest plain text or a Q&A entry directly (no file upload).",
    dependencies=[Depends(require_permission("data:manage"))],
)
def ingest_source(
    request: IngestRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[RagService, Depends(get_rag_service)],
) -> IngestResponse:
    try:
        result = service.ingest(
            workspace_id,
            IngestCommand(
                title=request.title,
                content=request.content,
                source_type=request.source_type,
                source_id=request.source_id,
                metadata=request.metadata,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EmbeddingProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return IngestResponse(
        document_id=result.document_id,
        title=result.title,
        source_type=result.source_type,
        chunk_count=result.chunk_count,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/rag/ingest/batch  — all three ModelConfigPage tabs at once
# ---------------------------------------------------------------------------

@router.post(
    "/ingest/batch",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Q&A pairs and knowledge entries in a single request.",
    dependencies=[Depends(require_permission("data:manage"))],
)
def ingest_batch(
    request: BatchIngestRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[RagService, Depends(get_rag_service)],
) -> BatchIngestResponse:
    results: list[IngestResponse] = []
    errors: list[str] = []

    # Q&A pairs — combine question + answer as the retrievable content.
    for qa in request.qa:
        content = f"Q: {qa.question.strip()}\nA: {qa.answer.strip()}"
        meta: dict = {"category": qa.category} if qa.category else {}
        try:
            r = service.ingest(
                workspace_id,
                IngestCommand(
                    title=qa.title,
                    content=content,
                    source_type=SourceType.QA,
                    metadata=meta,
                ),
            )
            results.append(IngestResponse(
                document_id=r.document_id,
                title=r.title,
                source_type=r.source_type,
                chunk_count=r.chunk_count,
            ))
        except ValueError as exc:
            errors.append(f"Q&A '{qa.title}': {exc}")
        except EmbeddingProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    # Knowledge entries.
    for kb in request.knowledge:
        meta = {"tags": kb.tags} if kb.tags else {}
        try:
            r = service.ingest(
                workspace_id,
                IngestCommand(
                    title=kb.title,
                    content=kb.content,
                    source_type=SourceType.KNOWLEDGE,
                    metadata=meta,
                ),
            )
            results.append(IngestResponse(
                document_id=r.document_id,
                title=r.title,
                source_type=r.source_type,
                chunk_count=r.chunk_count,
            ))
        except ValueError as exc:
            errors.append(f"Knowledge '{kb.title}': {exc}")
        except EmbeddingProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if errors and not results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(errors),
        )

    return BatchIngestResponse(ingested=len(results), results=results)


# ---------------------------------------------------------------------------
# POST /api/v1/rag/query
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline and receive a grounded answer with citations.",
    dependencies=[Depends(require_any_permission("dashboard:view", "monitoring:view", "prompt:use", "optimize:run"))],
)
def query_sources(
    request: QueryRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[RagService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryResponse:
    try:
        result = service.query(workspace_id, request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EmbeddingProviderError as exc:
        raise _embedding_http_error(exc) from exc
    except AnswerGeneratorError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    citations = [_citation_to_schema(match) for match in result.citations]
    embed_model, answer_model = _provider_models(settings)

    telemetry = None
    if result.telemetry is not None:
        telemetry = {
            "embed_ms": result.telemetry.embed_ms,
            "search_ms": result.telemetry.search_ms,
            "generate_ms": result.telemetry.generate_ms,
            "total_ms": result.telemetry.total_ms,
            "citation_count": result.telemetry.citation_count,
            "citation_coverage": result.telemetry.citation_coverage,
            "estimated_prompt_tokens": result.telemetry.estimated_prompt_tokens,
            "estimated_completion_tokens": result.telemetry.estimated_completion_tokens,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": embed_model,
            "answer_provider": settings.answer_provider,
            "answer_model": answer_model,
            "chunker": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            "retrieval": {"strategy": "dense", "min_score": settings.min_score},
        }

    return QueryResponse(
        answer=result.answer,
        trace_id=result.trace_id,
        citations=citations,
        telemetry=telemetry,
    )


@router.post(
    "/analyze",
    response_model=PolicyAnalysisResponse,
    summary="Analyze workspace evidence and return a chart-ready policy assessment.",
    dependencies=[Depends(require_permission("optimize:run"))],
)
def analyze_policy(
    request: PolicyAnalysisRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[RagService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PolicyAnalysisResponse:
    try:
        result = service.analyze(workspace_id, request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EmbeddingProviderError as exc:
        # Retrieval runs before generation, so an exhausted embedding quota used to
        # surface as a generic 502 and look like a bad model response. Report it as an
        # unavailable dependency and pass the real reason through.
        logger.exception("Policy analysis retrieval failed for workspace %s", workspace_id)
        raise _embedding_http_error(exc) from exc
    except (AnswerGeneratorError, PolicyAnalysisError) as exc:
        logger.exception("Policy analysis generation failed for workspace %s", workspace_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Policy analysis provider returned an invalid response: {exc}",
        ) from exc

    embed_model, configured_answer_model = _provider_models(settings)
    used_demo_fallback = result.analysis.get("analysis_basis") == "demo_template"
    answer_provider = "demo" if used_demo_fallback else settings.answer_provider
    answer_model = "demo" if used_demo_fallback else configured_answer_model
    citations = [_citation_to_schema(match) for match in result.citations]
    telemetry: dict | None = None
    if result.telemetry is not None:
        telemetry = {
            "embed_ms": result.telemetry.embed_ms,
            "search_ms": result.telemetry.search_ms,
            "generate_ms": result.telemetry.generate_ms,
            "total_ms": result.telemetry.total_ms,
            "citation_count": result.telemetry.citation_count,
            "citation_coverage": result.telemetry.citation_coverage,
            "estimated_prompt_tokens": result.telemetry.estimated_prompt_tokens,
            "estimated_completion_tokens": result.telemetry.estimated_completion_tokens,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": embed_model,
            "answer_provider": answer_provider,
            "answer_model": answer_model,
            "chunker": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            "retrieval": {"strategy": "dense", "min_score": settings.min_score},
        }

    return PolicyAnalysisResponse(
        **result.analysis,
        citations=citations,
        trace_id=result.trace_id,
        generated_at=utc_now(),
        provider=answer_provider,
        model=answer_model,
        telemetry=telemetry or {},
    )


# ---------------------------------------------------------------------------
# POST /api/v1/rag/scrape/preview  — fetch a URL, extract text, no ingest
# ---------------------------------------------------------------------------

@router.post(
    "/scrape/preview",
    response_model=ScrapePreviewResponse,
    summary="Fetch a public URL and return its readable text without ingesting it.",
    dependencies=[Depends(require_permission("data:manage"))],
)
def preview_scrape(
    request: ScrapePreviewRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ScrapePreviewResponse:
    url = request.url.strip()
    try:
        text, window_title = scrape_url(
            url,
            timeout_seconds=settings.scrape_timeout_seconds,
            max_bytes=settings.max_scrape_bytes,
        )
        text = validate_extracted_text(text, max_chars=settings.max_extracted_chars)
    except (ScrapeError, UploadValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ScrapePreviewResponse(
        url=url,
        window_title=window_title,
        suggested_title=_suggest_scrape_title(url, window_title),
        text=text,
        text_length=len(text),
    )


@router.post(
    "/prompt/optimize",
    response_model=PromptOptimizeResponse,
    summary="Rewrite a user prompt into a detailed, structured request using the LLM provider.",
    dependencies=[Depends(require_permission("prompt:use"))],
)
def optimize_prompt(
    request: PromptOptimizeRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PromptOptimizeResponse:
    generator = build_answer_generator(settings)
    try:
        optimized = generator.optimize_prompt(request.prompt)
    except (AnswerGeneratorError, EmbeddingProviderError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    optimized = (optimized or "").strip()
    if not optimized:
        optimized = request.prompt
    return PromptOptimizeResponse(optimized_prompt=optimized)
