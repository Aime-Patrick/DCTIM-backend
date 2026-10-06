"""Query intent classifier and router for DC-TIM RAG pipeline.

Classifies incoming user queries into analytical and retrieval intents:
- CONVERSATIONAL: greetings, pleasantries, assistant persona questions.
- FACTUAL_RAG: standard questions seeking grounded answers from workspace evidence.
- POLICY_ANALYSIS: what-if scenarios, policy lever adjustments, feasibility/impact simulations.
- KEYWORD_LOOKUP: specific acronyms, metric codes, indicator IDs, or section references.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryIntent(str, Enum):
    CONVERSATIONAL = "conversational"
    FACTUAL_RAG = "factual_rag"
    POLICY_ANALYSIS = "policy_analysis"
    KEYWORD_LOOKUP = "keyword_lookup"


# Acronym expansion lookup for national development policy domains.
ACRONYM_EXPANSIONS: dict[str, str] = {
    "gdp": "gross domestic product",
    "ndp": "national development plan",
    "sdg": "sustainable development goals",
    "kpi": "key performance indicator",
    "vat": "value added tax",
    "fdi": "foreign direct investment",
    "cpi": "consumer price index",
    "who": "world health organization",
    "unesco": "united nations educational scientific and cultural organization",
    "imf": "international monetary fund",
    "ev": "electric vehicle",
    "sme": "small and medium enterprises",
}

# Regex cues for conversational queries.
_GREETING_PATTERNS = (
    re.compile(r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|greetings|howdy)(?:\s+(?:there|all|dc-tim|assistant|friend))?[!.?]*$", re.IGNORECASE),
    re.compile(r"^(?:who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|help|thanks|thank\s+you)\b[!.?]*$", re.IGNORECASE),
)

# Regex cues for what-if scenarios and policy analysis interventions.
_SCENARIO_PATTERNS = (
    re.compile(r"\b(?:what\s+if|suppose|simulate|scenario\s*:|hypothetical)\b", re.IGNORECASE),
    re.compile(r"\b(?:increase|decrease|raise|reduce|cut|expand)\s+[\w\s]+\s+by\s+\d+", re.IGNORECASE),
    re.compile(r"\b(?:feasibility|likelihood|projected|projections|impact\s+of\s+increasing|outcome\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(?:target\s+value|policy\s+lever|intervention\s+scenario)\b", re.IGNORECASE),
)

# Regex cues for code/ID/acronym lookups.
_LOOKUP_PATTERNS = (
    re.compile(r"^(?:sdg|kpi|target|article|section|policy|kpi-)\s*[\d.]+\b", re.IGNORECASE),
    re.compile(r"^[A-Z0-9_\-]{2,10}(?:\s+[A-Z0-9_\-]{2,10})?$", re.ASCII),
)

# Category keyword detectors.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "education": ("education", "school", "teacher", "learning", "literacy", "enrolment", "curriculum", "university", "student"),
    "healthcare": ("health", "hospital", "doctor", "patient", "clinic", "disease", "vaccine", "mortality", "medical"),
    "economic": ("economic", "gdp", "trade", "fiscal", "employment", "investment", "tax", "inflation", "revenue", "budget"),
    "infrastructure": ("infrastructure", "road", "water", "energy", "transport", "housing", "grid", "sanitation", "broadband"),
    "governance": ("governance", "govern", "law", "transparency", "public sector", "corruption", "judiciary", "policy"),
    "environment": ("environment", "climate", "renewable", "forest", "emission", "biodiversity", "solar", "carbon", "pollution"),
}


@dataclass(frozen=True)
class IntentResult:
    intent: QueryIntent
    confidence: float
    category: str
    is_scenario: bool
    search_terms: list[str] = field(default_factory=list)
    suggested_action: str = "rag_retrieval"
    expanded_query: str = ""


def _extract_category(query: str) -> str:
    lower = query.lower()
    scores = {
        cat: sum(lower.count(kw) for kw in kws)
        for cat, kws in _CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _expand_query_terms(query: str) -> tuple[list[str], str]:
    """Tokenize query and inject known acronym expansions for lexical search."""
    tokens = re.findall(r"[\w\-]+", query)
    cleaned_terms: list[str] = []
    expansions: list[str] = []

    for t in tokens:
        cleaned_terms.append(t)
        t_lower = t.lower()
        if t_lower in ACRONYM_EXPANSIONS:
            expansions.append(ACRONYM_EXPANSIONS[t_lower])

    expanded_str = query
    if expansions:
        expanded_str = f"{query} {' '.join(expansions)}"

    return cleaned_terms, expanded_str


def classify_intent(query: str) -> IntentResult:
    """Classify user query intent and generate search enhancements."""
    cleaned = query.strip()
    if not cleaned:
        return IntentResult(
            intent=QueryIntent.CONVERSATIONAL,
            confidence=1.0,
            category="general",
            is_scenario=False,
            search_terms=[],
            suggested_action="direct_response",
            expanded_query="",
        )

    # 1. Check conversational greeting / meta
    for pattern in _GREETING_PATTERNS:
        if pattern.search(cleaned):
            return IntentResult(
                intent=QueryIntent.CONVERSATIONAL,
                confidence=0.95,
                category="general",
                is_scenario=False,
                search_terms=[],
                suggested_action="direct_response",
                expanded_query=cleaned,
            )

    # 2. Check what-if scenario / policy analysis
    for pattern in _SCENARIO_PATTERNS:
        if pattern.search(cleaned):
            category = _extract_category(cleaned)
            terms, expanded = _expand_query_terms(cleaned)
            return IntentResult(
                intent=QueryIntent.POLICY_ANALYSIS,
                confidence=0.90,
                category=category,
                is_scenario=True,
                search_terms=terms,
                suggested_action="policy_simulation",
                expanded_query=expanded,
            )

    # 3. Check code / ID lookup
    for pattern in _LOOKUP_PATTERNS:
        if pattern.search(cleaned) and len(cleaned) < 30:
            category = _extract_category(cleaned)
            terms, expanded = _expand_query_terms(cleaned)
            return IntentResult(
                intent=QueryIntent.KEYWORD_LOOKUP,
                confidence=0.88,
                category=category,
                is_scenario=False,
                search_terms=terms,
                suggested_action="keyword_search",
                expanded_query=expanded,
            )

    # 4. Standard Factual RAG
    category = _extract_category(cleaned)
    terms, expanded = _expand_query_terms(cleaned)
    return IntentResult(
        intent=QueryIntent.FACTUAL_RAG,
        confidence=0.85,
        category=category,
        is_scenario=False,
        search_terms=terms,
        suggested_action="rag_retrieval",
        expanded_query=expanded,
    )
