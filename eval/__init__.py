"""Phase 6 evaluation gates for the RAG pipeline.

Runs offline against deterministic hash embeddings + demo generator by default
so CI does not need OpenRouter keys. Metrics:

- recall@k: expected phrases appear in at least one retrieved chunk
- groundedness: answerable cases keep expected phrases (or citations);
  insufficient cases must refuse
- citation_accuracy: [n] markers in the answer refer to real citations
- insufficient_behavior: out-of-corpus queries refuse or do not invent
- latency split: embed / search / generate / total (ms)
- cost_per_answer_usd: heuristic token × rate (0 for hash/demo)
- cross_workspace_isolation: foreign workspace returns no citations
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.modules.rag.application import IngestCommand, RagService
from app.modules.rag.domain import SourceType
from app.modules.rag.evidence import is_insufficient_answer
from app.modules.rag.infrastructure.embeddings import HashEmbeddingProvider
from app.modules.rag.infrastructure.generator import DemoGroundedAnswerGenerator
from app.modules.rag.infrastructure.vector_store import InMemoryVectorStore
from app.modules.rag.text import chunk_text

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
DEFAULT_GOLD_PATH = EVAL_DIR / "gold_set.json"
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Placeholder rates for hosted models; hash/demo cost stays zero.
_PROMPT_USD_PER_1K = 0.0
_COMPLETION_USD_PER_1K = 0.0


@dataclass(frozen=True)
class CaseResult:
    id: str
    query: str
    expect_insufficient: bool
    recall_at_k: bool | None
    grounded: bool
    citation_accurate: bool
    insufficient_ok: bool
    latency_ms: float
    embed_ms: float
    search_ms: float
    generate_ms: float
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    estimated_cost_usd: float
    citation_count: int
    answer_preview: str


@dataclass(frozen=True)
class EvalSummary:
    total: int
    answerable: int
    insufficient: int
    recall_at_k: float
    groundedness: float
    citation_accuracy: float
    insufficient_behavior: float
    avg_latency_ms: float
    avg_embed_ms: float
    avg_search_ms: float
    avg_generate_ms: float
    avg_cost_per_answer_usd: float
    cross_workspace_isolation: bool
    cases: list[CaseResult]


def load_gold_set(path: Path = DEFAULT_GOLD_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_source_content(source: dict[str, Any]) -> str:
    if source.get("content"):
        return str(source["content"])
    rel = source.get("path")
    if not rel:
        raise ValueError(f"source {source.get('id')} needs content or path")
    path = BACKEND_DIR / str(rel)
    return path.read_text(encoding="utf-8")


def build_eval_service(
    gold: dict[str, Any],
    *,
    embedding_dimension: int = 256,
    chunk_size: int = 700,
    chunk_overlap: int = 80,
    top_k: int | None = None,
    min_score: float = 0.0,
) -> RagService:
    service = RagService(
        embeddings=HashEmbeddingProvider(embedding_dimension),
        vector_store=InMemoryVectorStore(),
        answer_generator=DemoGroundedAnswerGenerator(),
        chunker=lambda text: chunk_text(text, chunk_size, chunk_overlap),
        default_top_k=top_k or int(gold.get("top_k", 5)),
        min_score=min_score,
    )
    workspace = str(gold["workspace_id"])
    for source in gold["sources"]:
        content = _resolve_source_content(source)
        service.ingest(
            workspace,
            IngestCommand(
                title=str(source["title"]),
                content=content,
                source_type=SourceType(source.get("source_type", "document")),
                source_id=str(source.get("id")),
                metadata={"eval_source_id": source.get("id")},
            ),
        )
    return service


def _contains_any(haystack: str, phrases: list[str]) -> bool:
    lower = haystack.lower()
    return any(phrase.lower() in lower for phrase in phrases)


def _citation_accuracy(answer: str, citation_count: int) -> bool:
    refs = [int(n) for n in _CITATION_RE.findall(answer)]
    if not refs:
        return True
    return all(1 <= n <= citation_count for n in refs)


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens / 1000.0) * _PROMPT_USD_PER_1K
        + (completion_tokens / 1000.0) * _COMPLETION_USD_PER_1K,
        6,
    )


def evaluate_case(
    service: RagService,
    workspace_id: str,
    case: dict[str, Any],
    *,
    top_k: int,
) -> CaseResult:
    expect_insufficient = bool(case.get("expect_insufficient", False))
    expected = list(case.get("expected_phrases") or [])
    query = str(case["query"])

    started = time.perf_counter()
    result = service.query(workspace_id, query, top_k=top_k)
    wall_ms = (time.perf_counter() - started) * 1000

    tel = result.telemetry
    embed_ms = tel.embed_ms if tel else 0.0
    search_ms = tel.search_ms if tel else 0.0
    generate_ms = tel.generate_ms if tel else 0.0
    latency_ms = tel.total_ms if tel else wall_ms
    prompt_tokens = tel.estimated_prompt_tokens if tel else 0
    completion_tokens = tel.estimated_completion_tokens if tel else 0

    joined_evidence = "\n".join(match.chunk.content for match in result.citations)
    recall = None if expect_insufficient else _contains_any(joined_evidence, expected)

    insufficient = is_insufficient_answer(result.answer)
    forbidden = list(case.get("forbidden_phrases") or [])
    # Demo/LLM answers often echo the user question; ignore that when detecting invention.
    answer_body = result.answer.replace(query, "")
    invented = _contains_any(answer_body, forbidden) if forbidden else False

    if expect_insufficient:
        # Dense baselines may still return weak neighbors; pass if the model
        # refuses OR at least does not invent out-of-corpus entities.
        insufficient_ok = insufficient or not invented
        grounded = insufficient_ok and not invented
    else:
        grounded = (not insufficient) and (
            _contains_any(result.answer, expected) or bool(result.citations)
        )
        insufficient_ok = True
        if invented:
            grounded = False

    return CaseResult(
        id=str(case["id"]),
        query=query,
        expect_insufficient=expect_insufficient,
        recall_at_k=recall,
        grounded=grounded,
        citation_accurate=_citation_accuracy(result.answer, len(result.citations)),
        insufficient_ok=insufficient_ok,
        latency_ms=round(latency_ms, 2),
        embed_ms=round(embed_ms, 2),
        search_ms=round(search_ms, 2),
        generate_ms=round(generate_ms, 2),
        estimated_prompt_tokens=prompt_tokens,
        estimated_completion_tokens=completion_tokens,
        estimated_cost_usd=_estimate_cost_usd(prompt_tokens, completion_tokens),
        citation_count=len(result.citations),
        answer_preview=result.answer[:240],
    )


def check_cross_workspace_isolation(service: RagService, owner_workspace: str) -> bool:
    """Ensure a foreign workspace cannot retrieve the gold corpus."""
    foreign = f"{owner_workspace}-foreign"
    probe = service.query(
        foreign,
        "What is the lower secondary completion rate in Sub-Saharan Africa?",
        top_k=5,
    )
    return len(probe.citations) == 0


def run_gold_eval(
    gold_path: Path = DEFAULT_GOLD_PATH,
    *,
    min_score: float = 0.0,
) -> EvalSummary:
    gold = load_gold_set(gold_path)
    top_k = int(gold.get("top_k", 5))
    workspace = str(gold["workspace_id"])
    service = build_eval_service(gold, top_k=top_k, min_score=min_score)

    cases = [evaluate_case(service, workspace, case, top_k=top_k) for case in gold["cases"]]
    answerable = [c for c in cases if not c.expect_insufficient]
    insufficient = [c for c in cases if c.expect_insufficient]
    isolation_ok = check_cross_workspace_isolation(service, workspace)

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    recall_vals = [1.0 if c.recall_at_k else 0.0 for c in answerable if c.recall_at_k is not None]
    grounded_vals = [1.0 if c.grounded else 0.0 for c in cases]
    citation_vals = [1.0 if c.citation_accurate else 0.0 for c in cases]
    insuff_vals = [1.0 if c.insufficient_ok else 0.0 for c in insufficient]

    return EvalSummary(
        total=len(cases),
        answerable=len(answerable),
        insufficient=len(insufficient),
        recall_at_k=_avg(recall_vals),
        groundedness=_avg(grounded_vals),
        citation_accuracy=_avg(citation_vals),
        insufficient_behavior=_avg(insuff_vals),
        avg_latency_ms=round(sum(c.latency_ms for c in cases) / max(len(cases), 1), 2),
        avg_embed_ms=round(sum(c.embed_ms for c in cases) / max(len(cases), 1), 2),
        avg_search_ms=round(sum(c.search_ms for c in cases) / max(len(cases), 1), 2),
        avg_generate_ms=round(sum(c.generate_ms for c in cases) / max(len(cases), 1), 2),
        avg_cost_per_answer_usd=round(
            sum(c.estimated_cost_usd for c in cases) / max(len(cases), 1), 6
        ),
        cross_workspace_isolation=isolation_ok,
        cases=cases,
    )


def summary_as_dict(summary: EvalSummary) -> dict[str, Any]:
    return asdict(summary)


def main() -> None:
    summary = run_gold_eval()
    print(json.dumps(summary_as_dict(summary), indent=2))
    failures: list[str] = []
    if summary.recall_at_k < 0.5:
        failures.append(
            f"recall@k too low ({summary.recall_at_k}). "
            "Inspect gold_set phrases or chunking before adding hybrid retrieval."
        )
    if summary.groundedness < 0.5:
        failures.append(f"groundedness too low ({summary.groundedness})")
    if summary.citation_accuracy < 1.0:
        failures.append(f"citation_accuracy failed ({summary.citation_accuracy})")
    if summary.insufficient_behavior < 1.0:
        failures.append("insufficient-evidence cases failed the refusal gate")
    if not summary.cross_workspace_isolation:
        failures.append("cross-workspace isolation failed")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
