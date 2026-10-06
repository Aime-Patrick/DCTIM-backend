"""OpenAI-compatible embedding adapter.

Works with OpenAI, OpenRouter, Azure OpenAI, and other compatible gateways.
For ``text-embedding-3-*`` models the optional ``dimensions`` parameter is sent
so Matryoshka models can match the configured vector column size.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_BATCH_SIZE = 256


class EmbeddingProviderError(RuntimeError):
    """Raised when the remote embedding API fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _rate_limit_hint(response: httpx.Response) -> str | None:
    """Turn a gateway rate limit into an actionable, human-readable message.

    OpenRouter's free tier allows a fixed number of requests per day and reports when
    the allowance resets. Surfacing that turns an opaque 502 into something the operator
    can actually act on.
    """
    if response.status_code not in (429, 402):
        return None

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error = payload.get("error") if isinstance(payload, dict) else None
    message = str(error.get("message", "")) if isinstance(error, dict) else ""
    metadata = error.get("metadata") if isinstance(error, dict) else None
    headers = metadata.get("headers") if isinstance(metadata, dict) else {}
    headers = headers if isinstance(headers, dict) else {}

    limit = headers.get("X-RateLimit-Limit")
    remaining = headers.get("X-RateLimit-Remaining")
    reset = headers.get("X-RateLimit-Reset")
    source = metadata.get("limit_source") if isinstance(metadata, dict) else None

    parts = ["The embedding API rate limit was reached."]
    if source == "openrouter_free_tier_daily" or (limit and remaining == "0"):
        parts.append(
            "This is the OpenRouter free-tier daily allowance"
            + (f" of {limit} requests" if limit else "")
            + ", not a per-minute limit."
        )
    if reset:
        try:
            resets_at = datetime.fromtimestamp(int(reset) / 1000, tz=timezone.utc)
            parts.append(f"It resets at {resets_at:%Y-%m-%d %H:%M} UTC.")
        except (TypeError, ValueError):
            pass
    parts.append(
        "To fix now, either wait for the reset, add OpenRouter credits, or set "
        "RAG_EMBEDDING_PROVIDER=hash and re-upload the workspace documents so they are "
        "embedded locally with no provider quota."
    )
    hint = " ".join(parts)
    if message:
        logger.warning("Embedding rate limit detail: %s", message)
    return hint


class OpenAIEmbeddingProvider:
    """Hosted embedding provider implementing the EmbeddingProvider port."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimension: int = 256,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
        send_dimensions: bool | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if not api_key.strip():
            raise ValueError("API key is required for the hosted embedding provider")
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if batch_size <= 0 or batch_size > DEFAULT_EMBEDDING_BATCH_SIZE:
            raise ValueError(
                f"embedding batch_size must be between 1 and {DEFAULT_EMBEDDING_BATCH_SIZE}"
            )
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._extra_headers = dict(extra_headers or {})
        self._batch_size = batch_size
        # Auto: only OpenAI text-embedding-3 supports Matryoshka dimensions.
        if send_dimensions is None:
            send_dimensions = self._model.startswith("text-embedding-3") or (
                "/" in self._model and self._model.split("/", 1)[-1].startswith("text-embedding-3")
            )
        self._send_dimensions = send_dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                vectors: list[list[float]] = []
                for start in range(0, len(texts), self._batch_size):
                    batch = list(texts[start : start + self._batch_size])
                    vectors.extend(self._embed_batch(client, batch, headers))
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError(
                f"embedding request failed: {exc}"
            ) from exc
        return vectors

    def _embed_batch(
        self,
        client: httpx.Client,
        texts: list[str],
        headers: Mapping[str, str],
    ) -> list[list[float]]:
        payload: dict[str, object] = {
            "model": self._model,
            "input": texts,
        }
        if self._send_dimensions:
            payload["dimensions"] = self._dimension

        response = client.post(
            f"{self._base_url}/embeddings",
            headers=headers,
            json=payload,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            hint = _rate_limit_hint(response)
            message = f"embedding API returned {response.status_code}: {detail}"
            if hint:
                message = f"{message} | {hint}"
            raise EmbeddingProviderError(message, status_code=response.status_code)

        try:
            items = sorted(response.json()["data"], key=lambda row: row["index"])
            vectors = [list(map(float, row["embedding"])) for row in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError("unexpected embedding API response shape") from exc

        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                f"embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingProviderError(
                    f"embedding dimension mismatch: expected {self._dimension}, got {len(vector)}. "
                    "Set RAG_EMBEDDING_DIMENSION to match the model output "
                    "(OpenRouter nvidia/nemotron-3-embed-1b:free = 2048). "
                    "Re-ingest after changing dimension."
                )
        return vectors
