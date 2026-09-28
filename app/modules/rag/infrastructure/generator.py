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

    category = _demo_category(query, contexts)
    labels = {
        "education": "Education",
        "healthcare": "Healthcare",
        "economic": "Economic",
        "infrastructure": "Infrastructure",
        "governance": "Governance",
        "environment": "Environment",
        "general": "General policy",
    }
    subjects = {
        "education": "the education system",
        "healthcare": "the health system",
        "economic": "the economic sector",
        "infrastructure": "infrastructure delivery",
        "governance": "governance institutions",
        "environment": "environmental programmes",
        "general": "the policy area",
    }
    label = labels[category]
    subject = subjects[category]
    refs = [match.chunk.id for match in contexts[:3]]
    title = str(contexts[0].chunk.metadata.get("title", "")).strip()
    policy_name = sanitize_untrusted_text(title or query, max_chars=200) or "Policy analysis"
    excerpt = sanitize_untrusted_text(contexts[0].chunk.content, max_chars=320)
    confidence = min(0.65, round(0.35 + len(contexts) * 0.05, 2))
    return {
        "policy_name": policy_name,
        "category": category,
        "summary": (
            f"Demo template for {label.lower()} policy. Retrieved evidence mentions: {excerpt}. "
            "The following scores and actions are illustrative and should be validated against complete policy data."
        ),
        "evidence_status": "partial",
        "confidence": confidence,
        "analysis_basis": "demo_template",
        "feasibility": {
            "score": round(0.58 + min(len(contexts), 3) * 0.04, 2),
            "rationale": f"Demo template estimate for {subject}; implementation capacity is not fully established by the retrieved excerpts.",
        },
        "likelihood": {
            "score": round(0.54 + min(len(contexts), 3) * 0.04, 2),
            "rationale": f"Demo template estimate for {subject}; outcomes depend on conditions not covered by the retrieved excerpts.",
        },
        "risks": [
            {
                "title": f"{label} implementation capacity",
                "detail": f"The retrieved evidence may not cover the resources, responsibilities, or delivery constraints needed for {subject}.",
                "severity": "medium",
                "likelihood": 0.45,
                "impact": 0.55,
                "mitigation": f"Validate delivery responsibilities, funding, and milestones for {subject}.",
                "evidence_refs": refs,
            },
            {
                "title": "Evidence coverage gap",
                "detail": "The available excerpts are limited and do not establish a complete baseline for the policy.",
                "severity": "low",
                "likelihood": 0.4,
                "impact": 0.4,
                "mitigation": "Add current implementation reports, outcome data, and evaluation findings to the workspace.",
                "evidence_refs": refs[:1],
            },
        ],
        "recommendations": [
            {
                "title": f"Validate the {label.lower()} baseline",
                "detail": f"Confirm the current status, targets, and ownership measures for {subject} before making allocation decisions.",
                "priority": "high",
                "expected_impact": "Creates a traceable baseline for implementation monitoring.",
                "confidence": 0.45,
                "timeframe": "0–3 months",
                "evidence_refs": refs,
            },
            {
                "title": f"Strengthen {label.lower()} monitoring",
                "detail": f"Track a small set of outcome and delivery indicators for {subject} and review them at regular intervals.",
                "priority": "medium",
                "expected_impact": "Improves early detection of delivery gaps.",
                "confidence": 0.4,
                "timeframe": "3–12 months",
                "evidence_refs": refs[:1],
            },
        ],
        "metrics": [
            {
                "label": f"{label} outcome indicator",
                "baseline": None,
                "projected": None,
                "change_percent": None,
                "unit": "index",
                "direction": "neutral",
                "confidence": 0.25,
                "rationale": "Demo template leaves numeric values null because the retrieved evidence does not provide a verified baseline and projection.",
                "evidence_refs": refs[:1],
            }
        ],
        "dimensions": [
            {
                "label": "Evidence coverage",
                "score": min(0.75, 0.35 + len(contexts) * 0.08),
                "rationale": "Illustrative score based on the number of retrieved excerpts.",
            },
            {
                "label": "Implementation readiness",
                "score": 0.5,
                "rationale": f"Demo template score for {subject}; delivery readiness is not confirmed by the excerpts.",
            },
            {
                "label": "Outcome measurability",
                "score": 0.4,
                "rationale": "Numeric outcomes should be confirmed with additional workspace evidence.",
            },
        ],
        "phases": [
            {
                "label": "Evidence consolidation",
                "time_horizon": "0–3 months",
                "progress": 0.2,
                "actions": [
                    f"Collect authoritative documents about {subject}.",
                    "Confirm baseline indicators and implementation owners.",
                ],
            },
            {
                "label": "Implementation review",
                "time_horizon": "3–12 months",
                "progress": 0.35,
                "actions": [
                    "Review delivery milestones and risks.",
                    "Update recommendations using measured outcomes.",
                ],
            },
        ],
        "uncertainties": [
            "The retrieved excerpts do not establish a complete policy baseline.",
            "Projected outcomes are not estimated in demo mode.",
        ],
        "next_steps": [
            f"Ingest current implementation and outcome evidence for {subject}.",
            "Re-run analysis after the additional evidence is indexed.",
        ],
    }


class DemoGroundedAnswerGenerator:
    """Evidence-preserving fallback until a real LLM provider is selected."""

    def generate(self, query: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return INSUFFICIENT_EVIDENCE_MESSAGE
        evidence = "\n".join(f"- {match.chunk.content}" for match in contexts)
        return (
            "Demo mode: the answer generator is not connected to an LLM yet. "
            f"Relevant evidence for '{query}':\n{evidence}"
        )

    def analyze(self, query: str, contexts: Sequence[RetrievedChunk]) -> dict[str, object]:
        return _demo_analysis(query, contexts)

    def optimize_prompt(self, prompt: str) -> str:
        return prompt.strip()


class FallbackAnswerGenerator:
    def __init__(self, primary: AnswerGenerator, fallback: AnswerGenerator) -> None:
        self._primary = primary
        self._fallback = fallback

    def generate(self, query: str, contexts: Sequence[RetrievedChunk]) -> str:
        try:
            return self._primary.generate(query, contexts)
        except AnswerGeneratorError:
            return self._fallback.generate(query, contexts)

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
