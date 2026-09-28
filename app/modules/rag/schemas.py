from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .domain import SourceType


# ---------------------------------------------------------------------------
# Text / Q&A ingest (JSON body)
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    source_type: SourceType = SourceType.DOCUMENT
    source_id: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    title: str
    source_type: SourceType
    chunk_count: int
    file_path: str | None = None


# ---------------------------------------------------------------------------
# File upload (multipart — returned after saving + ingesting a file)
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    document_id: str
    title: str
    filename: str
    file_path: str
    file_url: str | None = None
    source_type: SourceType
    chunk_count: int


# ---------------------------------------------------------------------------
# Batch ingest (ModelConfigPage submits all tabs at once)
# ---------------------------------------------------------------------------

class QAPairRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    category: str | None = None


class KnowledgeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class BatchIngestRequest(BaseModel):
    """Receives all three tabs from ModelConfigPage in one request."""
    model_name: str = Field(default="", max_length=100)
    qa: list[QAPairRequest] = Field(default_factory=list)
    knowledge: list[KnowledgeRequest] = Field(default_factory=list)


class BatchIngestResponse(BaseModel):
    ingested: int
    results: list[IngestResponse]


# ---------------------------------------------------------------------------
# Listing (Data Lake view)
# ---------------------------------------------------------------------------

class DataEntry(BaseModel):
    id: str
    title: str
    source_type: SourceType
    source_id: str | None = None
    created_at: datetime
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int = 0
    preview: str | None = None
    tags: list[str] = Field(default_factory=list)
    category: str | None = None


class DataSummaryResponse(BaseModel):
    total_entries: int
    total_chunks: int
    count_by_type: dict[str, int] = Field(default_factory=dict)


class DataListResponse(BaseModel):
    summary: DataSummaryResponse
    total: int
    entries: list[DataEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source_type: SourceType
    text: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    trace_id: str
    citations: list[CitationResponse]
    telemetry: dict[str, Any] | None = None


class PolicyAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NullableScore = Annotated[float | None, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
NullableFiniteNumber = Annotated[float | None, Field(allow_inf_nan=False)]
EvidenceRef = Annotated[str, Field(min_length=1, max_length=200)]
ActionText = Annotated[str, Field(min_length=1, max_length=500)]


class _AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FeasibilityAnalysis(_AnalysisModel):
    score: NullableScore = None
    rationale: str = Field(min_length=1, max_length=2000)


class LikelihoodAnalysis(_AnalysisModel):
    score: NullableScore = None
    rationale: str = Field(min_length=1, max_length=2000)


class PolicyRisk(_AnalysisModel):
    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(min_length=1, max_length=3000)
    severity: Literal["low", "medium", "high", "critical"]
    likelihood: Score
    impact: Score
    mitigation: str = Field(min_length=1, max_length=3000)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=50)


class PolicyRecommendation(_AnalysisModel):
    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(min_length=1, max_length=3000)
    priority: Literal["low", "medium", "high", "critical"]
    expected_impact: str = Field(min_length=1, max_length=2000)
    confidence: Score
    timeframe: str = Field(min_length=1, max_length=200)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=50)


class PolicyMetric(_AnalysisModel):
    label: str = Field(min_length=1, max_length=200)
    baseline: NullableFiniteNumber = None
    projected: NullableFiniteNumber = None
    change_percent: NullableFiniteNumber = None
    unit: str = Field(min_length=1, max_length=50)
    direction: Literal["increase", "decrease", "neutral"]
    confidence: Score
    rationale: str = Field(min_length=1, max_length=2000)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _calculate_change_percent(self) -> PolicyMetric:
        if self.baseline is not None and self.projected is not None and self.baseline != 0:
            change = ((self.projected - self.baseline) / self.baseline) * 100
            if not isfinite(change):
                raise ValueError("change_percent must be finite")
            self.change_percent = round(change, 6)
        else:
            self.change_percent = None
        return self


class PolicyDimension(_AnalysisModel):
    label: str = Field(min_length=1, max_length=200)
    score: Score
    rationale: str = Field(min_length=1, max_length=2000)


class PolicyPhase(_AnalysisModel):
    label: str = Field(min_length=1, max_length=200)
    time_horizon: str = Field(min_length=1, max_length=200)
    progress: Score
    actions: list[ActionText] = Field(default_factory=list, max_length=20)


class PolicyAnalysisContent(_AnalysisModel):
    policy_name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=6000)
    evidence_status: Literal["sufficient", "partial", "insufficient"]
    confidence: Score
    analysis_basis: str = Field(min_length=1, max_length=100)
    feasibility: FeasibilityAnalysis
    likelihood: LikelihoodAnalysis
    risks: list[PolicyRisk] = Field(default_factory=list, max_length=20)
    recommendations: list[PolicyRecommendation] = Field(default_factory=list, max_length=20)
    metrics: list[PolicyMetric] = Field(default_factory=list, max_length=30)
    dimensions: list[PolicyDimension] = Field(default_factory=list, max_length=20)
    phases: list[PolicyPhase] = Field(default_factory=list, max_length=10)
    uncertainties: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        default_factory=list, max_length=20
    )
    next_steps: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        default_factory=list, max_length=20
    )

    @field_validator("analysis_basis", mode="before")
    @classmethod
    def limit_analysis_basis_length(cls, value: Any) -> Any:
        # Providers sometimes return an explanatory sentence here even though this
        # field is a short label. Keep the API contract bounded without failing the
        # whole analysis for that harmless overrun.
        return value[:100] if isinstance(value, str) else value


class PolicyAnalysisResponse(PolicyAnalysisContent):
    citations: list[CitationResponse] = Field(default_factory=list, max_length=50)
    trace_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    telemetry: dict[str, Any] = Field(default_factory=dict)


AnalysisRequest = PolicyAnalysisRequest
AnalyzeRequest = PolicyAnalysisRequest
FeasibilityResponse = FeasibilityAnalysis
LikelihoodResponse = LikelihoodAnalysis
RiskResponse = PolicyRisk
RecommendationResponse = PolicyRecommendation
MetricResponse = PolicyMetric
DimensionResponse = PolicyDimension
PhaseResponse = PolicyPhase
PolicyAnalysis = PolicyAnalysisContent


# ---------------------------------------------------------------------------
# Web scraping (fetch + extract text; NOT ingested by this endpoint)
# ---------------------------------------------------------------------------

class ScrapePreviewRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class ScrapePreviewResponse(BaseModel):
    url: str
    window_title: str | None = None
    suggested_title: str
    text: str
    text_length: int


# ---------------------------------------------------------------------------
# Prompt optimization (LLM rewrites a user prompt; no RAG evidence)
# ---------------------------------------------------------------------------

class PromptOptimizeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=50_000)

    @model_validator(mode="after")
    def _reject_blank(self) -> PromptOptimizeRequest:
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        return self


class PromptOptimizeResponse(BaseModel):
    optimized_prompt: str
