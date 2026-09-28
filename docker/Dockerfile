# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Keep Python from buffering stdout/stderr and generating .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies layer — rebuilt only when pyproject.toml changes
# ---------------------------------------------------------------------------
FROM base AS deps

COPY pyproject.toml ./
# Install runtime deps only (no dev extras in the image)
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Runtime image
# ---------------------------------------------------------------------------
FROM base AS runtime

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Run Alembic migrations then start the server.
# Using a shell entrypoint keeps the migration step visible in logs and makes
# it easy to skip migrations in one-off containers (override CMD).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
