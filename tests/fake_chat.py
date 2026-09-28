"""Test doubles for the OpenAI-compatible chat transport.

``OpenAIGroundedAnswerGenerator`` streams the response body so it can enforce a
wall-clock deadline (an ``httpx`` read timeout alone only bounds a single socket
read, which a slow token stream can satisfy indefinitely).  Fakes must therefore
implement the streaming surface of ``httpx.Client`` -- ``stream()`` yielding a
response with ``read()`` and ``iter_bytes()`` -- not just ``post()``.
"""
from __future__ import annotations

import json as _json
from collections.abc import Callable, Mapping
from typing import Any


class FakeStreamResponse:
    """Minimal stand-in for a streaming ``httpx`` response."""

    def __init__(
        self,
        status_code: int,
        body: Mapping[str, Any] | None = None,
        *,
        raw: bytes | None = None,
        chunk_delay: float = 0.0,
    ) -> None:
        self.status_code = status_code
        self._raw = raw if raw is not None else _json.dumps(dict(body or {})).encode("utf-8")
        self._chunk_delay = chunk_delay

    @property
    def text(self) -> str:
        return self._raw.decode("utf-8", "replace")

    def read(self) -> bytes:
        return self._raw

    def iter_bytes(self):
        # Split in two so callers that poll the clock between chunks are
        # genuinely exercised rather than reading everything in one shot.
        midpoint = max(1, len(self._raw) // 2)
        if self._chunk_delay:
            import time

            time.sleep(self._chunk_delay)
        yield self._raw[:midpoint]
        if self._chunk_delay:
            import time

            time.sleep(self._chunk_delay)
        yield self._raw[midpoint:]


class _StreamContext:
    def __init__(self, response: FakeStreamResponse) -> None:
        self._response = response

    def __enter__(self) -> FakeStreamResponse:
        return self._response

    def __exit__(self, *args: Any) -> None:
        return None


class FakeChatClient:
    """Callable replacement for ``httpx.Client`` inside the generator module."""

    def __init__(self, handler: Callable[[Mapping[str, Any]], FakeStreamResponse]) -> None:
        self._handler = handler
        self.requests: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> "FakeChatClient":
        return self

    def __enter__(self) -> "FakeChatClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def stream(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
    ) -> _StreamContext:
        assert method == "POST", f"unexpected HTTP method: {method}"
        assert url.endswith("/chat/completions"), f"unexpected URL: {url}"
        self.requests.append({"url": url, "headers": dict(headers or {}), "json": json})
        return _StreamContext(self._handler(json))


def chat_content(content: str) -> dict[str, Any]:
    """Build a successful chat-completion body containing *content*."""
    return {"choices": [{"message": {"content": content}}]}
