"""Simple in-process sliding-window rate limiter for API routes."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class SlidingWindowRateLimiter:
    """Per-key request budget over a fixed window (seconds)."""

    def __init__(self, *, limit: int, window_seconds: int = 60) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._limit:
                return False
            bucket.append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limits to /api/v1 routes (health excluded)."""

    def __init__(self, app, *, limit_per_minute: int) -> None:
        super().__init__(app)
        self._limiter = SlidingWindowRateLimiter(limit=limit_per_minute, window_seconds=60)

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if not path.startswith("/api/v1"):
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        auth = request.headers.get("authorization", "")
        key = f"{client_host}:{auth[:32]}" if auth else client_host
        if not self._limiter.allow(key):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again in a minute."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
