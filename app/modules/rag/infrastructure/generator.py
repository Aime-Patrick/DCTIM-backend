from __future__ import annotations

from collections.abc import Sequence

from ..domain import RetrievedChunk
from ..evidence import INSUFFICIENT_EVIDENCE_MESSAGE, sanitize_untrusted_text
from ..ports import AnswerGenerator
from .openai_generator import AnswerGeneratorError, ProviderUnavailableError


_DEMO_CATEGORY_KEYWORDS = {
    "education": ("education", "school", "teacher", "learning", "literacy", "enrolment"),
    "healthcare": ("health", "hospital", "doctor", "patient", "clinic", "disease"),
    "economic": ("economic", "gdp", "trade", "fiscal", "employment", "investment"),
    "infrastructure": ("infrastructure", "road", "water", "energy", "transport", "housing"),
    "governance": ("governance", "govern", "law", "transparency", "public sector", "corruption"),
    "environment": ("environment", "climate", "renewable", "forest", "emission", "biodiversity"),
}


def _demo_category(query: str, contexts: Sequence[RetrievedChunk]) -> str:
    haystack = " ".join(
        [query, *(str(match.chunk.metadata.get("title", "")) for match in contexts), *(match.chunk.content for match in contexts)]
    ).lower()
    scores = {
        category: sum(haystack.count(keyword) for keyword in keywords)
        for category, keywords in _DEMO_CATEGORY_KEYWORDS.items()
    }
    category = max(scores, key=scores.get)
    return category if scores[category] else "general"


def _demo_analysis(query: str, contexts: Sequence[RetrievedChunk]) -> dict[str, object]:
    if not contexts:
        return {
            "policy_name": query[:200] or "Policy analysis",
            "category": "general",
            "summary": INSUFFICIENT_EVIDENCE_MESSAGE,
            "evidence_status": "insufficient",
            "confidence": 0.0,
            "analysis_basis": "insufficient_evidence",
            "feasibility": {
                "score": None,
                "rationale": "No relevant evidence was retrieved for this workspace.",
            },
            "likelihood": {
                "score": None,
                "rationale": "No relevant evidence was retrieved for this workspace.",
            },
            "risks": [],
            "recommendations": [],
            "metrics": [],
            "dimensions": [],
            "phases": [],
            "uncertainties": [],
            "next_steps": [],
        }

    title = str(contexts[0].chunk.metadata.get("title", "")).strip()
    policy_name = sanitize_untrusted_text(title or query, max_chars=200) or "Policy analysis"
    excerpt = sanitize_untrusted_text(contexts[0].chunk.content, max_chars=320)
    category = _demo_category(query, contexts)
    return {
        "policy_name": policy_name,
        "category": category,
        "summary": (
            "Strict source-only mode is active. No model analysis was generated; "
            f"the indexed evidence begins with: {excerpt}"
        ),
        "evidence_status": "partial",
        "confidence": 0.0,
        "analysis_basis": "evidence_only_no_llm",
        "feasibility": {
            "score": None,
            "rationale": "No feasibility score is produced without a source-constrained analysis model.",
        },
        "likelihood": {
            "score": None,
            "rationale": "No likelihood score is produced without a source-constrained analysis model.",
        },
        "risks": [],
        "recommendations": [],
        "metrics": [],
        "dimensions": [],
        "phases": [],
        "uncertainties": ["No source-only analysis model is configured."],
        "next_steps": [],
    }


class DemoGroundedAnswerGenerator:
    """Evidence-preserving fallback until a real LLM provider is selected."""

    answer_mode = "evidence_only"

    def generate(self, query: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return INSUFFICIENT_EVIDENCE_MESSAGE
        evidence = "\n".join(
            f"- {match.chunk.content} [{index}]"
            for index, match in enumerate(contexts, start=1)
        )
        return (
            "Demo mode: the answer generator is not connected to an LLM yet. "
            f"Relevant evidence for '{query}':\n{evidence}"
        )

    def analyze(self, query: str, contexts: Sequence[RetrievedChunk]) -> dict[str, object]:
        return _demo_analysis(query, contexts)

    def optimize_prompt(self, prompt: str) -> str:
        return prompt.strip()


class EvidenceOnlyAnswerGenerator(DemoGroundedAnswerGenerator):
    """Safe degraded mode when every hosted synthesis route is unavailable.

    This returns retrieved workspace excerpts and citations only. It never adds
    facts from the model's general knowledge, so provider outages degrade answer
    quality without degrading grounding guarantees.
    """

    def generate(self, query: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return INSUFFICIENT_EVIDENCE_MESSAGE
        excerpts = []
        for index, match in enumerate(contexts, start=1):
            title = sanitize_untrusted_text(
                str(match.chunk.metadata.get("title", "Workspace source")),
                max_chars=200,
            )
            excerpt = sanitize_untrusted_text(match.chunk.content, max_chars=2500)
            excerpts.append(f"[{index}] {title}\n{excerpt}")
        return (
            "I could not generate a synthesized answer right now. "
            "Here are the relevant workspace excerpts:\n\n"
            + "\n\n".join(excerpts)
        )


class FallbackAnswerGenerator:
    def __init__(self, primary: AnswerGenerator, fallback: AnswerGenerator) -> None:
        self._primary = primary
        self._fallback = fallback
        self.last_answer_mode = "synthesized"

    def generate(self, query: str, contexts: Sequence[RetrievedChunk]) -> str:
        try:
            answer = self._primary.generate(query, contexts)
            self.last_answer_mode = getattr(self._primary, "answer_mode", "synthesized")
            return answer
        except AnswerGeneratorError:
            answer = self._fallback.generate(query, contexts)
            self.last_answer_mode = getattr(self._fallback, "answer_mode", "evidence_only")
            return answer

    def analyze(self, query: str, contexts: Sequence[RetrievedChunk]) -> dict[str, object]:
        try:
            return self._primary.analyze(query, contexts)
        except ProviderUnavailableError:
            return self._fallback.analyze(query, contexts)

    def optimize_prompt(self, prompt: str) -> str:
        try:
            return self._primary.optimize_prompt(prompt)
        except ProviderUnavailableError:
            return self._fallback.optimize_prompt(prompt)


class ChainedAnswerGenerator:
    """Try configured providers in order, advancing when one is unavailable."""

    def __init__(self, providers: Sequence[AnswerGenerator]) -> None:
        if not providers:
            raise ValueError("at least one answer provider is required")
        self._providers = tuple(providers)

    def _run(self, method: str, *args: object) -> object:
        errors: list[str] = []
        for provider in self._providers:
            try:
                return getattr(provider, method)(*args)
            except AnswerGeneratorError as exc:
                errors.append(str(exc))
        message = "; ".join(errors) or "all configured providers failed"
        raise ProviderUnavailableError(message)

    def generate(self, query: str, contexts: Sequence[RetrievedChunk]) -> str:
        return self._run("generate", query, contexts)  # type: ignore[return-value]

    def analyze(self, query: str, contexts: Sequence[RetrievedChunk]) -> dict[str, object]:
        return self._run("analyze", query, contexts)  # type: ignore[return-value]

    def optimize_prompt(self, prompt: str) -> str:
        return self._run("optimize_prompt", prompt)  # type: ignore[return-value]
