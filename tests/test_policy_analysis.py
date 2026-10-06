from __future__ import annotations

import os

from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.auth.users import get_user_store
from app.modules.rag.domain import Chunk, RetrievedChunk
from app.modules.rag.infrastructure.generator import DemoGroundedAnswerGenerator
from app.modules.rag.infrastructure.openai_generator import OpenAIGroundedAnswerGenerator
from tests.fake_chat import FakeChatClient, FakeStreamResponse, chat_content


def _client() -> TestClient:
    get_user_store.cache_clear()
    deps.get_settings.cache_clear()
    deps.get_auth_service.cache_clear()
    deps._memory_vector_store = None
    deps._engine = None
    deps._session_factory = None
    os.environ.pop("DATABASE_URL", None)
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hash"
    os.environ["RAG_ANSWER_PROVIDER"] = "demo"
    return TestClient(create_app())


def _login(client: TestClient) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    )
    return response.json()["access_token"]


def test_analyze_requires_authentication() -> None:
    client = _client()
    response = client.post(
        "/api/v1/rag/analyze",
        json={"query": "education policy", "top_k": 3},
    )
    assert response.status_code == 401


def test_demo_analyze_returns_grounded_analysis_after_ingest() -> None:
    client = _client()
    token = _login(client)
    headers = {"Authorization": f"Bearer {token}"}
    ingest = client.post(
        "/api/v1/rag/ingest",
        headers=headers,
        json={
            "title": "Education policy",
            "content": "Teacher training improves learning outcomes and school quality.",
            "source_type": "knowledge",
        },
    )
    assert ingest.status_code == 201

    response = client.post(
        "/api/v1/rag/analyze",
        headers=headers,
        json={"query": "What improves learning outcomes?", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "demo"
    assert body["model"] == "demo"
    assert body["analysis_basis"] == "evidence_only_no_llm"
    assert body["category"] == "education"
    assert body["evidence_status"] in {"sufficient", "partial", "insufficient"}
    assert body["citations"]
    assert body["risks"] == []
    assert body["recommendations"] == []
    assert body["metrics"] == []
    assert body["dimensions"] == []
    assert body["phases"] == []
    assert body["uncertainties"]
    assert body["next_steps"] == []
    assert body["trace_id"]
    assert body["generated_at"]
    assert body["telemetry"]["citation_count"] == len(body["citations"])


def test_analyze_without_evidence_is_insufficient_and_skips_generator(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("analysis generator must not be called without evidence")

    monkeypatch.setattr(DemoGroundedAnswerGenerator, "analyze", fail_if_called)
    client = _client()
    token = _login(client)
    response = client.post(
        "/api/v1/rag/analyze",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "unrelated policy question"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_status"] == "insufficient"
    assert body["citations"] == []
    for key in ("risks", "recommendations", "metrics", "dimensions", "phases", "uncertainties", "next_steps"):
        assert body[key] == []


def test_openai_analyze_parses_fenced_json_and_requests_json(monkeypatch) -> None:
    fenced = '```json\n{"policy_name":"Education policy","category":"education"}\n```'
    client = FakeChatClient(lambda payload: FakeStreamResponse(200, chat_content(fenced)))

    monkeypatch.setattr(
        "app.modules.rag.infrastructure.openai_generator.httpx.Client",
        client,
    )
    generator = OpenAIGroundedAnswerGenerator("test-key", model="test-model")
    context = RetrievedChunk(
        chunk=Chunk(
            id="chunk-1",
            document_id="document-1",
            workspace_id="workspace",
            content="Teacher training improves learning outcomes.",
            ordinal=0,
            metadata={"title": "Education policy", "source_type": "knowledge"},
        ),
        score=0.9,
    )

    result = generator.analyze("What improves learning outcomes?", [context])

    assert result["policy_name"] == "Education policy"
    assert result["category"] == "education"
    captured = client.requests[0]
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert "<<<EVIDENCE" in captured["json"]["messages"][1]["content"]
