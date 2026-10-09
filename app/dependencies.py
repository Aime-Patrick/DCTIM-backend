"""FastAPI dependency providers.

Wiring strategy
---------------
- If ``DATABASE_URL`` is set, the app uses the persistent stack:
    PgVectorStore + DocumentRepository backed by PostgreSQL/pgvector.
- If ``DATABASE_URL`` is absent (unit tests, quick local runs without Docker),
    the app falls back to the in-memory stack so the process still starts.
- Embedding / answer providers are selected from Settings:
    hash/demo for local tests, openrouter (free) when ``OPENROUTER_API_KEY`` is set.
- Auth uses JWT bearer tokens; workspace is derived from verified claims.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .core.config import Settings
from .modules.auth.domain import AuthUser
from .modules.auth.service import AuthService
from .modules.auth.users import get_user_store
from .modules.rag.application import RagService
from .modules.rag.infrastructure.embeddings import HashEmbeddingProvider
from .modules.rag.infrastructure.generator import (
    ChainedAnswerGenerator,
    DemoGroundedAnswerGenerator,
    EvidenceOnlyAnswerGenerator,
    FallbackAnswerGenerator,
)
from .modules.rag.infrastructure.vector_store import InMemoryVectorStore
from .modules.rag.ports import AnswerGenerator, EmbeddingProvider
from .modules.rag.text import chunk_text

_bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Settings (singleton)
# ---------------------------------------------------------------------------

@lru_cache
def get_settings() -> Settings:
    return Settings.from_environment()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@lru_cache
def get_auth_service() -> AuthService:
    settings = get_settings()
    if _ensure_db_configured():
        from .modules.auth.repository import UserRepository

        users = UserRepository(_session_factory)
    else:
        users = get_user_store()
    return AuthService(
        users,
        jwt_secret=settings.jwt_secret,
        jwt_expires_minutes=settings.jwt_expires_minutes,
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    auth: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization Bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return auth.user_from_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_workspace_id(
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> str:
    """Workspace is always taken from the verified JWT — never from the client."""
    workspace_id = user.workspace_id.strip()
    if not workspace_id or len(workspace_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user is missing a valid workspace",
        )
    return workspace_id


def has_permission(user: AuthUser, permission: str) -> bool:
    return permission in user.permissions


def require_permission(permission: str):
    def dependency(user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUser:
        if not has_permission(user, permission):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission required: {permission}")
        return user
    return dependency


def require_any_permission(*permissions: str):
    def dependency(user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUser:
        if not any(has_permission(user, permission) for permission in permissions):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Required workspace permission is missing.")
        return user
    return dependency


# ---------------------------------------------------------------------------
# Database engine / session factory (created once, only when DATABASE_URL set)
# ---------------------------------------------------------------------------

_engine = None
_session_factory = None


def _ensure_db_configured() -> bool:
    """Initialise the engine/session factory if DATABASE_URL is now available.

    Returns True if the persistent stack is ready, False otherwise.
    """
    global _engine, _session_factory
    if _session_factory is not None:
        return True

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return False

    from .modules.rag.infrastructure.db.session import build_engine, build_session_factory

    _engine = build_engine(database_url)
    _session_factory = build_session_factory(_engine)
    return True


def _get_db_session() -> Generator:
    """Yield a per-request database session, or None when no DB is configured."""
    if not _ensure_db_configured():
        yield None
        return

    from sqlalchemy.orm import Session

    session: Session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------

def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "RAG_EMBEDDING_PROVIDER=openrouter requires OPENROUTER_API_KEY to be set"
            )
        from .modules.rag.infrastructure.openai_embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            settings.openrouter_api_key,
            model=settings.openrouter_embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            base_url=settings.openrouter_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            send_dimensions=False,
            extra_headers={
                "HTTP-Referer": settings.openrouter_http_referer,
                "X-Title": settings.openrouter_app_title,
            },
        )

    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "RAG_EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set"
            )
        from .modules.rag.infrastructure.openai_embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    return HashEmbeddingProvider(settings.embedding_dimension)


def build_answer_generator(settings: Settings) -> AnswerGenerator:
    if settings.answer_provider == "demo":
        return DemoGroundedAnswerGenerator()

    from .modules.rag.infrastructure.openai_generator import OpenAIGroundedAnswerGenerator

    provider_configs = {
        "openrouter": (
            settings.openrouter_api_key,
            settings.openrouter_base_url,
            settings.openrouter_chat_model,
            settings.openrouter_chat_fallbacks,
            settings.openrouter_chat_max_tokens,
            settings.openrouter_analysis_max_tokens,
            {
                "HTTP-Referer": settings.openrouter_http_referer,
                "X-Title": settings.openrouter_app_title,
            },
        ),
        "openai": (
            settings.openai_api_key,
            settings.openai_base_url,
            settings.openai_chat_model,
            (),
            16384,
            16384,
            {},
        ),
        "gemini": (
            settings.gemini_api_key,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            settings.gemini_chat_model,
            (),
            16384,
            16384,
            {},
        ),
        "groq": (
            settings.groq_api_key,
            "https://api.groq.com/openai/v1",
            settings.groq_chat_model,
            (),
            settings.openrouter_chat_max_tokens,
            settings.openrouter_analysis_max_tokens,
            {},
        ),
        "nvidia": (
            settings.nvidia_api_key,
            "https://integrate.api.nvidia.com/v1",
            settings.nvidia_chat_model,
            settings.nvidia_chat_fallbacks,
            settings.openrouter_chat_max_tokens,
            settings.openrouter_analysis_max_tokens,
            {},
        ),
    }
    order = (settings.answer_provider,) + tuple(
        name for name in provider_configs if name != settings.answer_provider
    )
    providers = []
    for name in order:
        api_key, base_url, model, fallbacks, max_tokens, analysis_max_tokens, headers = provider_configs[name]
        if not api_key or not model.strip():
            continue
        providers.append(
            OpenAIGroundedAnswerGenerator(
                api_key,
                model=model,
                fallback_models=fallbacks,
                base_url=base_url,
                timeout_seconds=settings.answer_timeout_seconds,
                max_tokens=max_tokens,
                analysis_max_tokens=analysis_max_tokens,
                extra_headers=headers,
            )
        )

    if not providers:
        return DemoGroundedAnswerGenerator()
    # If every hosted route is unavailable, return retrieved excerpts rather
    # than a generic or pretrained answer. This keeps the application useful
    # while preserving the source-only grounding contract.
    return FallbackAnswerGenerator(
        ChainedAnswerGenerator(providers),
        EvidenceOnlyAnswerGenerator(),
    )


# ---------------------------------------------------------------------------
# RagService dependency
# ---------------------------------------------------------------------------

_memory_vector_store: InMemoryVectorStore | None = None


def _get_memory_vector_store() -> InMemoryVectorStore:
    """Process-wide in-memory store so ingest survives across requests without Postgres."""
    global _memory_vector_store
    if _memory_vector_store is None:
        _memory_vector_store = InMemoryVectorStore()
    return _memory_vector_store


def get_rag_service(
    settings: Annotated[Settings, Depends(get_settings)],
    db_session: Annotated[object, Depends(_get_db_session)],
) -> RagService:
    """Build a RagService wired for the current environment."""
    embeddings = build_embedding_provider(settings)
    answer_generator = build_answer_generator(settings)
    chunker = lambda text: chunk_text(text, settings.chunk_size, settings.chunk_overlap)  # noqa: E731

    if db_session is not None:
        from sqlalchemy.orm import Session as _Session

        from .modules.rag.infrastructure.db.repository import DocumentRepository
        from .modules.rag.infrastructure.pg_vector_store import PgVectorStore

        session: _Session = db_session  # type: ignore[assignment]
        return RagService(
            embeddings=embeddings,
            vector_store=PgVectorStore(session),
            answer_generator=answer_generator,
            chunker=chunker,
            document_repository=DocumentRepository(session),
            default_top_k=settings.default_top_k,
            min_score=settings.min_score,
        )

    return RagService(
        embeddings=embeddings,
        vector_store=_get_memory_vector_store(),
        answer_generator=answer_generator,
        chunker=chunker,
        document_repository=None,
        default_top_k=settings.default_top_k,
        min_score=settings.min_score,
    )


# ---------------------------------------------------------------------------
# ChatService dependency (per-user conversation history)
# ---------------------------------------------------------------------------

_memory_chat_store = None


def _get_memory_chat_store():
    """Process-wide in-memory chat store so history survives requests without Postgres."""
    global _memory_chat_store
    if _memory_chat_store is None:
        from .modules.chat.infrastructure.memory_store import InMemoryChatStore

        _memory_chat_store = InMemoryChatStore()
    return _memory_chat_store


def get_chat_service(
    db_session: Annotated[object, Depends(_get_db_session)],
):
    """Build a chat store wired for the current environment."""
    from .modules.chat.application import ChatService

    if db_session is not None:
        from sqlalchemy.orm import Session as _Session

        from .modules.chat.infrastructure.repository import ChatRepository

        session: _Session = db_session  # type: ignore[assignment]
        return ChatService(ChatRepository(session))

    return ChatService(_get_memory_chat_store())


_memory_policy_store = None


def _get_memory_policy_store():
    global _memory_policy_store
    if _memory_policy_store is None:
        from .modules.policies.infrastructure.memory_store import InMemoryPolicyStore

        _memory_policy_store = InMemoryPolicyStore()
    return _memory_policy_store


def get_policy_service(
    db_session: Annotated[object, Depends(_get_db_session)],
):
    from .modules.policies.application import PolicyService

    if db_session is not None:
        from .modules.policies.infrastructure.repository import PolicyRepository

        session = db_session
        return PolicyService(PolicyRepository(session))

    return PolicyService(_get_memory_policy_store())


_memory_case_store = None


def _get_memory_case_store():
    global _memory_case_store
    if _memory_case_store is None:
        from .modules.cases.infrastructure.memory_store import InMemoryTransformationCaseStore

        _memory_case_store = InMemoryTransformationCaseStore()
    return _memory_case_store


def get_case_service(
    db_session: Annotated[object, Depends(_get_db_session)],
):
    from .modules.cases.application import TransformationCaseService

    if db_session is not None:
        from .modules.cases.infrastructure.repository import TransformationCaseRepository

        return TransformationCaseService(TransformationCaseRepository(db_session))

    return TransformationCaseService(_get_memory_case_store())


_memory_indicator_store = None


def _get_memory_indicator_store():
    global _memory_indicator_store
    if _memory_indicator_store is None:
        from .modules.indicators.infrastructure.memory_store import InMemoryIndicatorStore

        _memory_indicator_store = InMemoryIndicatorStore()
    return _memory_indicator_store


def get_indicator_service(
    db_session: Annotated[object, Depends(_get_db_session)],
):
    from .modules.indicators.application import IndicatorService

    if db_session is not None:
        from .modules.indicators.infrastructure.repository import IndicatorRepository

        return IndicatorService(IndicatorRepository(db_session))

    return IndicatorService(_get_memory_indicator_store())


_memory_intervention_store = None


def _get_memory_intervention_store():
    global _memory_intervention_store
    if _memory_intervention_store is None:
        from .modules.interventions.infrastructure.memory_store import InMemoryInterventionStore

        _memory_intervention_store = InMemoryInterventionStore()
    return _memory_intervention_store


def get_intervention_service(
    db_session: Annotated[object, Depends(_get_db_session)],
):
    from .modules.interventions.application import InterventionService

    if db_session is not None:
        from .modules.interventions.infrastructure.repository import InterventionRepository

        return InterventionService(InterventionRepository(db_session))

    return InterventionService(_get_memory_intervention_store())
