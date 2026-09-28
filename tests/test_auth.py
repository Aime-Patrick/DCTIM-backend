"""Auth module tests: login, JWT workspace identity, cross-workspace isolation."""
from __future__ import annotations

import os

from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.auth.users import get_user_store, workspace_from_email


def _client() -> TestClient:
    get_user_store.cache_clear()
    deps.get_settings.cache_clear()
    deps.get_auth_service.cache_clear()
    deps._memory_vector_store = None
    deps._engine = None
    deps._session_factory = None
    # Avoid hanging on an unreachable local Postgres from the developer's .env.
    os.environ.pop("DATABASE_URL", None)
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hash"
    os.environ["RAG_ANSWER_PROVIDER"] = "demo"
    return TestClient(create_app())


def test_workspace_from_email_slug() -> None:
    assert workspace_from_email("Admin@dc-tim.ai") == "admin"
    assert workspace_from_email("policy.analyst@example.com") == "policy-analyst"


def test_login_returns_jwt_and_workspace() -> None:
    client = _client()
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "admin@dc-tim.ai"
    assert body["user"]["workspace_id"] == "admin"
    assert body["access_token"]


def test_login_rejects_bad_password() -> None:
    client = _client()
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "wrong"},
    )
    assert response.status_code == 401


def test_rag_requires_bearer_token() -> None:
    client = _client()
    response = client.post(
        "/api/v1/rag/query",
        json={"query": "hello", "top_k": 3},
        headers={"X-Workspace-Id": "admin"},
    )
    assert response.status_code == 401


def test_rag_entries_requires_bearer_token() -> None:
    client = _client()
    response = client.get("/api/v1/rag/entries")
    assert response.status_code == 401


def test_rag_entries_returns_empty_listing_without_db() -> None:
    client = _client()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    ).json()
    response = client.get(
        "/api/v1/rag/entries",
        params={"source_type": "knowledge", "search": "teacher"},
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["entries"] == []
    assert body["summary"] == {
        "total_entries": 0,
        "total_chunks": 0,
        "count_by_type": {},
    }


def test_me_returns_authenticated_user() -> None:
    client = _client()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "analyst@dc-tim.ai", "password": "analyst123"},
    ).json()
    me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["workspace_id"] == "analyst"


def test_jwt_workspace_isolates_rag_data() -> None:
    client = _client()

    admin = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    ).json()
    analyst = client.post(
        "/api/v1/auth/login",
        json={"email": "analyst@dc-tim.ai", "password": "analyst123"},
    ).json()

    ingest = client.post(
        "/api/v1/rag/ingest",
        headers={"Authorization": f"Bearer {admin['access_token']}"},
        json={
            "title": "Admin only",
            "content": "Secret admin evidence about teacher training outcomes.",
            "source_type": "knowledge",
        },
    )
    assert ingest.status_code == 201

    admin_query = client.post(
        "/api/v1/rag/query",
        headers={"Authorization": f"Bearer {admin['access_token']}"},
        json={"query": "teacher training", "top_k": 3},
    )
    assert admin_query.status_code == 200
    assert admin_query.json()["citations"]

    analyst_query = client.post(
        "/api/v1/rag/query",
        headers={"Authorization": f"Bearer {analyst['access_token']}"},
        json={"query": "teacher training", "top_k": 3},
    )
    assert analyst_query.status_code == 200
    assert analyst_query.json()["citations"] == []
    assert "could not find relevant evidence" in analyst_query.json()["answer"].lower()


def test_scrape_preview_requires_bearer_token() -> None:
    client = _client()
    response = client.post(
        "/api/v1/rag/scrape/preview",
        json={"url": "https://example.com/page"},
        headers={"X-Workspace-Id": "admin"},
    )
    assert response.status_code == 401


def test_scrape_preview_validates_url_scheme() -> None:
    client = _client()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    ).json()
    response = client.post(
        "/api/v1/rag/scrape/preview",
        json={"url": "ftp://example.com/page"},
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )
    assert response.status_code == 422
    assert "http(s)" in response.json()["detail"]


def test_scrape_preview_fetches_and_extracts(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        content = (
            b"<html><head><title>Health Policy</title></head>"
            b"<body><script>ignore()</script><p>Free clinics open nationwide.</p></body></html>"
        )

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "app.modules.rag.scraper.httpx.Client",
        FakeClient,
    )

    client = _client()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    ).json()
    response = client.post(
        "/api/v1/rag/scrape/preview",
        json={"url": "https://example.com/health"},
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == "https://example.com/health"
    assert body["window_title"] == "Health Policy"
    assert body["suggested_title"] == "Health Policy"
    assert "Free clinics open nationwide" in body["text"]
    assert "ignore()" not in body["text"]
    assert body["text_length"] == len(body["text"])


def test_prompt_optimize_requires_bearer_token() -> None:
    client = _client()
    response = client.post(
        "/api/v1/rag/prompt/optimize",
        json={"prompt": "GDP growth drivers"},
        headers={"X-Workspace-Id": "admin"},
    )
    assert response.status_code == 401


def test_prompt_optimize_rejects_empty_prompt() -> None:
    client = _client()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    ).json()
    response = client.post(
        "/api/v1/rag/prompt/optimize",
        json={"prompt": "   "},
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )
    assert response.status_code == 422


def test_prompt_optimize_in_demo_mode_echoes_prompt() -> None:
    client = _client()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    ).json()
    response = client.post(
        "/api/v1/rag/prompt/optimize",
        json={"prompt": "What drives GDP growth in the health sector?"},
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )
    assert response.status_code == 200
    assert response.json()["optimized_prompt"] == "What drives GDP growth in the health sector?"
