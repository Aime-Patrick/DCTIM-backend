from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before anything reads os.environ (uvicorn does not do this).
# Values already in the environment (e.g. set by Docker) are NOT overwritten.
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import Settings
from .core.rate_limit import RateLimitMiddleware
from .modules.auth.api import router as auth_router
from .modules.chat.api import router as chat_router
from .modules.policies.api import router as policies_router
from .modules.rag.api import router as rag_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings.from_environment()
    app = FastAPI(
        title="DC-TIM RAG Backend",
        version="0.1.0",
        description="Modular monolith API for policy knowledge ingestion and grounded retrieval.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        limit_per_minute=runtime_settings.rate_limit_per_minute,
    )
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(policies_router, prefix="/api/v1")
    app.include_router(rag_router, prefix="/api/v1")

    @app.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "service": app.title,
            "version": app.version,
            "description": app.description,
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
        }

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": runtime_settings.environment}

    @app.on_event("startup")
    def _check_db() -> None:
        """Verify the database is reachable on startup (when DATABASE_URL is set)."""
        from .dependencies import _ensure_db_configured

        if not _ensure_db_configured():
            logger.warning(
                "DATABASE_URL is not set — running with in-memory storage. "
                "Data will not persist across restarts."
            )
            return

        try:
            from .dependencies import _engine
            from .modules.rag.infrastructure.db.session import check_db_connection

            check_db_connection(_engine)
            logger.info("Database connection verified.")
        except Exception as exc:  # pragma: no cover
            logger.error("Could not connect to the database: %s", exc)
            raise

    return app


app = create_app()
