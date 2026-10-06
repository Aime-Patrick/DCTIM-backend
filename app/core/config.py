from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _optional_str(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _csv_models(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    """Runtime settings kept separate from feature code."""

    environment: str
    cors_origins: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    embedding_dimension: int
    default_top_k: int
    min_score: float
    upload_dir: str
    storage_provider: str
    cloudinary_cloud_name: str | None
    cloudinary_api_key: str | None
    cloudinary_api_secret: str | None
    cloudinary_folder: str
    cloudinary_timeout_seconds: float
    embedding_provider: str
    answer_provider: str
    openai_api_key: str | None
    openai_base_url: str
    openai_embedding_model: str
    openai_chat_model: str
    openai_timeout_seconds: float
    answer_timeout_seconds: float
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_embedding_model: str
    openrouter_chat_model: str
    openrouter_chat_fallbacks: tuple[str, ...]
    openrouter_chat_max_tokens: int
    openrouter_analysis_max_tokens: int
    openrouter_http_referer: str
    openrouter_app_title: str
    gemini_api_key: str | None
    gemini_chat_model: str
    groq_api_key: str | None
    groq_chat_model: str
    nvidia_api_key: str | None
    nvidia_chat_model: str
    nvidia_chat_fallbacks: tuple[str, ...]
    jwt_secret: str
    jwt_expires_minutes: int
    rate_limit_per_minute: int
    max_upload_bytes: int
    max_extracted_chars: int
    scrape_timeout_seconds: float
    max_scrape_bytes: int

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv("RAG_CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        )
        chunk_size = _int_env("RAG_CHUNK_SIZE", 1000)
        chunk_overlap = _int_env("RAG_CHUNK_OVERLAP", 120)
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be >= 0 and smaller than RAG_CHUNK_SIZE")
        embedding_batch_size = _int_env("RAG_EMBEDDING_BATCH_SIZE", 256)
        if embedding_batch_size <= 0 or embedding_batch_size > 256:
            raise ValueError("RAG_EMBEDDING_BATCH_SIZE must be between 1 and 256")

        openai_key = _optional_str("OPENAI_API_KEY")
        openrouter_key = _optional_str("OPENROUTER_API_KEY")
        gemini_key = _optional_str("GEMINI_API_KEY")
        groq_key = _optional_str("GROQ_API_KEY")
        nvidia_key = _optional_str("NVIDIA_API_KEY")
        embedding_provider = os.getenv("RAG_EMBEDDING_PROVIDER", "").strip().lower()
        answer_provider = os.getenv("RAG_ANSWER_PROVIDER", "").strip().lower()

        # Prefer free OpenRouter embeddings when a key is present.
        if not embedding_provider:
            if openrouter_key:
                embedding_provider = "openrouter"
            elif openai_key:
                embedding_provider = "openai"
            else:
                embedding_provider = "hash"
        if not answer_provider:
            if openrouter_key:
                answer_provider = "openrouter"
            elif openai_key:
                answer_provider = "openai"
            elif gemini_key:
                answer_provider = "gemini"
            elif groq_key:
                answer_provider = "groq"
            elif nvidia_key:
                answer_provider = "nvidia"
            else:
                answer_provider = "demo"

        # Dev safety: do not hard-fail the whole app when a key is missing.
        if embedding_provider == "openrouter" and not openrouter_key:
            embedding_provider = "hash"
        if embedding_provider == "openai" and not openai_key:
            embedding_provider = "hash"
        configured_keys = {
            "openrouter": openrouter_key,
            "openai": openai_key,
            "gemini": gemini_key,
            "groq": groq_key,
            "nvidia": nvidia_key,
        }
        if answer_provider != "demo" and not configured_keys.get(answer_provider):
            answer_provider = next(
                (name for name in ("openrouter", "openai", "gemini", "groq", "nvidia") if configured_keys[name]),
                "demo",
            )

        if embedding_provider not in {"hash", "openai", "openrouter"}:
            raise ValueError(
                "RAG_EMBEDDING_PROVIDER must be 'hash', 'openai', or 'openrouter'"
            )
        if answer_provider not in {"demo", "openai", "openrouter", "gemini", "groq", "nvidia"}:
            raise ValueError(
                "RAG_ANSWER_PROVIDER must be 'demo', 'openai', 'openrouter', 'gemini', 'groq', or 'nvidia'"
            )

        timeout_raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "60").strip()
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be a number") from exc
        if timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be positive")

        answer_timeout_raw = os.getenv("RAG_ANSWER_TIMEOUT_SECONDS", "15").strip()
        try:
            answer_timeout_seconds = float(answer_timeout_raw)
        except ValueError as exc:
            raise ValueError("RAG_ANSWER_TIMEOUT_SECONDS must be a number") from exc
        if answer_timeout_seconds <= 0:
            raise ValueError("RAG_ANSWER_TIMEOUT_SECONDS must be positive")

        min_score_raw = os.getenv("RAG_MIN_SCORE", "0.15").strip()
        try:
            min_score = float(min_score_raw)
        except ValueError as exc:
            raise ValueError("RAG_MIN_SCORE must be a number") from exc
        if min_score < 0 or min_score > 1:
            raise ValueError("RAG_MIN_SCORE must be between 0 and 1")

        environment = os.getenv("RAG_ENVIRONMENT", "development")

        cloudinary_cloud_name = _optional_str("CLOUDINARY_CLOUD_NAME")
        cloudinary_api_key = _optional_str("CLOUDINARY_API_KEY")
        cloudinary_api_secret = _optional_str("CLOUDINARY_API_SECRET")
        cloudinary_folder = (
            os.getenv("CLOUDINARY_FOLDER", "dc-tim/uploads").strip() or "dc-tim/uploads"
        )
        cloudinary_timeout_raw = os.getenv("CLOUDINARY_TIMEOUT_SECONDS", "60").strip()
        try:
            cloudinary_timeout = float(cloudinary_timeout_raw)
        except ValueError as exc:
            raise ValueError("CLOUDINARY_TIMEOUT_SECONDS must be a number") from exc
        if cloudinary_timeout <= 0:
            raise ValueError("CLOUDINARY_TIMEOUT_SECONDS must be positive")
        cloudinary_complete = bool(
            cloudinary_cloud_name and cloudinary_api_key and cloudinary_api_secret
        )

        storage_provider = os.getenv("RAG_STORAGE_PROVIDER", "").strip().lower()
        if not storage_provider:
            storage_provider = "cloudinary" if cloudinary_complete else "local"
        if storage_provider not in {"local", "cloudinary"}:
            raise ValueError("RAG_STORAGE_PROVIDER must be 'local' or 'cloudinary'")

        # Falling back to local disk on a serverless runtime silently loses every
        # uploaded file on the next cold start, so fail loudly in production.
        if storage_provider == "cloudinary" and not cloudinary_complete:
            if environment == "production":
                raise ValueError(
                    "RAG_STORAGE_PROVIDER=cloudinary requires CLOUDINARY_CLOUD_NAME, "
                    "CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET to be set"
                )
            storage_provider = "local"

        jwt_secret = os.getenv("AUTH_JWT_SECRET", "").strip()
        if not jwt_secret:
            if environment == "production":
                raise ValueError("AUTH_JWT_SECRET is required when RAG_ENVIRONMENT=production")
            jwt_secret = "dev-only-change-me-dc-tim-jwt-secret"

        jwt_expires = _int_env("AUTH_JWT_EXPIRES_MINUTES", 60 * 12)
        if jwt_expires <= 0:
            raise ValueError("AUTH_JWT_EXPIRES_MINUTES must be positive")

        rate_limit = _int_env("RAG_RATE_LIMIT_PER_MINUTE", 120)
        if rate_limit <= 0:
            raise ValueError("RAG_RATE_LIMIT_PER_MINUTE must be positive")

        max_upload_mb = _int_env("RAG_MAX_UPLOAD_MB", 20)
        if max_upload_mb <= 0:
            raise ValueError("RAG_MAX_UPLOAD_MB must be positive")

        max_extracted = _int_env("RAG_MAX_EXTRACTED_CHARS", 500_000)
        if max_extracted <= 0:
            raise ValueError("RAG_MAX_EXTRACTED_CHARS must be positive")

        scrape_timeout_raw = os.getenv("RAG_SCRAPE_TIMEOUT_SECONDS", "20").strip()
        try:
            scrape_timeout = float(scrape_timeout_raw)
        except ValueError as exc:
            raise ValueError("RAG_SCRAPE_TIMEOUT_SECONDS must be a number") from exc
        if scrape_timeout <= 0:
            raise ValueError("RAG_SCRAPE_TIMEOUT_SECONDS must be positive")

        max_scrape_mb = _int_env("RAG_MAX_SCRAPE_MB", 5)
        if max_scrape_mb <= 0:
            raise ValueError("RAG_MAX_SCRAPE_MB must be positive")

        # Default dim matches OpenRouter nvidia/nemotron-3-embed-1b:free (2048).
        # Hash/openai users can override via RAG_EMBEDDING_DIMENSION.
        default_dim = 2048 if embedding_provider == "openrouter" else 256

        return cls(
            environment=environment,
            cors_origins=origins,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_batch_size=embedding_batch_size,
            embedding_dimension=_int_env("RAG_EMBEDDING_DIMENSION", default_dim),
            default_top_k=_int_env("RAG_DEFAULT_TOP_K", 5),
            min_score=min_score,
            upload_dir=os.getenv("RAG_UPLOAD_DIR", ""),
            storage_provider=storage_provider,
            cloudinary_cloud_name=cloudinary_cloud_name,
            cloudinary_api_key=cloudinary_api_key,
            cloudinary_api_secret=cloudinary_api_secret,
            cloudinary_folder=cloudinary_folder,
            cloudinary_timeout_seconds=cloudinary_timeout,
            embedding_provider=embedding_provider,
            answer_provider=answer_provider,
            openai_api_key=openai_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            openai_timeout_seconds=timeout_seconds,
            answer_timeout_seconds=answer_timeout_seconds,
            openrouter_api_key=openrouter_key,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            openrouter_embedding_model=os.getenv(
                "OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"
            ),
            openrouter_chat_model=os.getenv(
                "OPENROUTER_CHAT_MODEL", "qwen/qwen3.8-27b:free"
            ),
            openrouter_chat_fallbacks=_csv_models(
                os.getenv(
                    "OPENROUTER_CHAT_FALLBACKS",
                    "nvidia/nemotron-3.5-lightning:free,"
                    "google/gemma-4-26b-a4b-it:free,"
                    "deepseek/deepseek-v4-flash-0731:free",
                )
            ),
            openrouter_chat_max_tokens=_int_env("OPENROUTER_CHAT_MAX_TOKENS", 4096),
            openrouter_analysis_max_tokens=_int_env("OPENROUTER_ANALYSIS_MAX_TOKENS", 8192),
            openrouter_http_referer=os.getenv(
                "OPENROUTER_HTTP_REFERER", "http://localhost:5173"
            ),
            openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE", "DC-TIM"),
            gemini_api_key=gemini_key,
            gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"),
            groq_api_key=groq_key,
            groq_chat_model=os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b"),
            nvidia_api_key=nvidia_key,
            # Do not assume a provider model that may be retired. Set this only
            # when an NVIDIA route is intentionally enabled.
            nvidia_chat_model=os.getenv("NVIDIA_CHAT_MODEL", "").strip(),
            nvidia_chat_fallbacks=_csv_models(
                os.getenv(
                    "NVIDIA_CHAT_FALLBACKS",
                    "nvidia/nemotron-3-super-49b-v1,"
                    "nvidia/nemotron-3.5-lightning",
                )
            ),
            jwt_secret=jwt_secret,
            jwt_expires_minutes=jwt_expires,
            rate_limit_per_minute=rate_limit,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            max_extracted_chars=max_extracted,
            scrape_timeout_seconds=scrape_timeout,
            max_scrape_bytes=max_scrape_mb * 1024 * 1024,
        )
