"""Pytest coverage for Phase 6 evaluation gates."""
from __future__ import annotations

from app.modules.rag.evidence import filter_by_min_score, is_insufficient_answer
from app.modules.rag.domain import Chunk, RetrievedChunk
from eval import run_gold_eval


def test_filter_by_min_score_drops_weak_hits() -> None:
    contexts = [
        RetrievedChunk(
            chunk=Chunk(
                id="a",
                document_id="d",
                workspace_id="w",
                content="strong",
                ordinal=0,
            ),
            score=0.8,
        ),
        RetrievedChunk(
            chunk=Chunk(
                id="b",
                document_id="d",
                workspace_id="w",
                content="weak",
                ordinal=1,
            ),
            score=0.05,
        ),
    ]
    kept = filter_by_min_score(contexts, 0.2)
    assert len(kept) == 1
    assert kept[0].chunk.id == "a"


def test_is_insufficient_answer_detects_refusal() -> None:
    assert is_insufficient_answer("I could not find relevant evidence in the indexed workspace.")
    assert not is_insufficient_answer("Teacher training improves learning outcomes [1].")


def test_gold_set_eval_meets_baseline_gates() -> None:
    summary = run_gold_eval(min_score=0.0)
    assert summary.total >= 10
    assert summary.recall_at_k >= 0.5
    assert summary.groundedness >= 0.5
    assert summary.citation_accuracy == 1.0
    assert summary.insufficient_behavior == 1.0
    assert summary.cross_workspace_isolation is True
    assert summary.avg_latency_ms >= 0.0
    assert summary.avg_embed_ms >= 0.0
    assert summary.avg_search_ms >= 0.0
    assert summary.avg_generate_ms >= 0.0
    assert summary.avg_cost_per_answer_usd == 0.0
    assert all(c.estimated_prompt_tokens >= 0 for c in summary.cases)
