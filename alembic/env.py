from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Load .env file so `alembic upgrade head` works from the shell without
# manually exporting DATABASE_URL first.
# ---------------------------------------------------------------------------
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ---------------------------------------------------------------------------
# Import our SQLAlchemy metadata so Alembic can detect schema changes
# ---------------------------------------------------------------------------
from app.modules.rag.infrastructure.db.models import Base  # noqa: E402
import app.modules.chat.infrastructure.models as _chat_models  # noqa: E402,F401
import app.modules.policies.infrastructure.models as _policy_models
import app.modules.auth.models as _auth_models
import app.modules.cases.infrastructure.models as _case_models  # noqa: E402,F401
import app.modules.indicators.infrastructure.models as _indicator_models  # noqa: E402,F401
import app.modules.interventions.infrastructure.models as _intervention_models  # noqa: E402,F401

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Resolve the database URL (env var wins over alembic.ini)
# ---------------------------------------------------------------------------
_database_url = os.environ.get("DATABASE_URL")
if not _database_url:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Copy .env.example to .env and fill in your credentials."
    )


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Generate SQL without a live connection (for review / dry-run)."""
    context.configure(
        url=_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = create_engine(_database_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
