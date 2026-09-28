from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence


class HashEmbeddingProvider:
    """Deterministic local embedding adapter for development and tests.

    This is intentionally not presented as a production-quality semantic model. A hosted
    embedding adapter can implement the same port when the project selects its provider.
    """

    _token_pattern = re.compile(r"[\w-]+", re.UNICODE)

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = self._token_pattern.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]

