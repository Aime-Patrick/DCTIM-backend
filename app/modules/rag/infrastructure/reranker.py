"""Re-ranking and score fusion for RAG retrieval.

Implements:
- Reciprocal Rank Fusion (RRF) for blending dense vector and lexical BM25 results.
- Metadata relevance weighting (exact title/source type matching).
- Document diversity de-duplication to prevent over-representing a single document.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from ..domain import Chunk, RetrievedChunk


class Reranker(Protocol):
    """Contract for re-ranking retrieved chunks."""

    def rerank(
        self,
        query: str,
        contexts: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...


def reciprocal_rank_fusion(
    dense_hits: Sequence[RetrievedChunk],
    lexical_hits: Sequence[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit.chunk.id
        scores[cid] += 1.0 / (k + rank)
        chunks[cid] = hit.chunk

    for rank, hit in enumerate(lexical_hits, start=1):
        cid = hit.chunk.id
        scores[cid] += 1.0 / (k + rank)
        if cid not in chunks:
            chunks[cid] = hit.chunk

    if not scores:
        return []

    # Normalize scores to 0..1 scale based on max theoretical RRF score
    max_score = 2.0 / (k + 1)  # If ranked #1 in both lists
    fused: list[RetrievedChunk] = []
    for cid, raw_score in scores.items():
        norm_score = min(1.0, round(raw_score / max_score, 6))
        fused.append(RetrievedChunk(chunk=chunks[cid], score=norm_score))

    fused.sort(key=lambda item: item.score, reverse=True)
    return fused


class HybridRRFReRanker:
    """Reranker that applies query-term overlap, metadata boosts, and diversity penalty."""

    def __init__(
        self,
        title_weight: float = 0.15,
        exact_term_weight: float = 0.10,
        diversity_decay: float = 0.85,
    ) -> None:
        self._title_weight = title_weight
        self._exact_term_weight = exact_term_weight
        self._diversity_decay = diversity_decay

    def rerank(
        self,
        query: str,
        contexts: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not contexts or top_k <= 0:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        doc_frequency: dict[str, int] = defaultdict(int)
        reranked: list[RetrievedChunk] = []

        for match in contexts:
            base_score = match.score
            chunk = match.chunk
            title = str(chunk.metadata.get("title", "")).lower()
            content = chunk.content.lower()

            # 1. Title keyword overlap boost
            title_tokens = set(re.findall(r"\w+", title))
            title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
            title_boost = title_overlap * self._title_weight

            # 2. Exact phrase / term frequency boost in content
            content_tokens = set(re.findall(r"\w+", content))
            content_overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            term_boost = content_overlap * self._exact_term_weight

            # 3. Document diversity penalty (discount if same doc already has top hits)
            seen_count = doc_frequency[chunk.document_id]
            diversity_multiplier = math.pow(self._diversity_decay, seen_count)
            doc_frequency[chunk.document_id] += 1

            final_score = (base_score + title_boost + term_boost) * diversity_multiplier
            clamped_score = min(1.0, max(0.0, round(final_score, 6)))

            reranked.append(RetrievedChunk(chunk=chunk, score=clamped_score))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
