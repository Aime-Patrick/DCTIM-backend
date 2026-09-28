from __future__ import annotations

import os

from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.auth.users import get_user_store


def _client() -> TestClient:
    get_user_store.cache_clear()
    deps.get_settings.cache_clear()
    deps.get_auth_service.cache_clear()
    deps._memory_policy_store = None
    deps._engine = None
    deps._session_factory = None
    os.environ.pop("DATABASE_URL", None)
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hash"
    os.environ["RAG_ANSWER_PROVIDER"] = "demo"
    return TestClient(create_app())


def _login(client: TestClient, email: str, password: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _payload() -> dict[str, object]:
    return {
        "title": "Education policy",
        "content": "Teacher training improves learning outcomes.",
        "category": "education",
        "analysis_trace_id": "trace-1",
        "analysis_provider": "demo",
        "analysis": {
            "policy_name": "Education policy",
            "summary": "Structured analysis",
            "risks": [],
        },
    }


def test_policy_crud_requires_authentication() -> None:
    client = _client()
    assert client.get("/api/v1/policies").status_code == 401
    assert client.post("/api/v1/policies", json=_payload()).status_code == 401


def test_policy_crud_and_revision() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")
    headers = _auth(token)

    created = client.post("/api/v1/policies", json=_payload(), headers=headers)
    assert created.status_code == 201
    body = created.json()
    policy_id = body["id"]
    assert body["revision"] == 1
    assert body["analysis_trace_id"] == "trace-1"
    assert body["analysis"]["summary"] == "Structured analysis"
    assert body["has_analysis"] is True

    listed = client.get("/api/v1/policies", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert "content" not in listed.json()["policies"][0]

    updated = client.patch(
        f"/api/v1/policies/{policy_id}",
        json={"title": "Updated education policy", "content": "Updated content.", "category": "schools"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["title"] == "Updated education policy"
    assert updated.json()["analysis_provider"] == "demo"
    assert updated.json()["analysis"]["summary"] == "Structured analysis"

    detail = client.get(f"/api/v1/policies/{policy_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["content"] == "Updated content."


def test_policy_workspace_isolation() -> None:
    client = _client()
    admin = _login(client, "admin@dc-tim.ai", "admin123")
    analyst = _login(client, "analyst@dc-tim.ai", "analyst123")
    created = client.post(
        "/api/v1/policies",
        json=_payload(),
        headers=_auth(admin),
    ).json()

    assert client.get("/api/v1/policies", headers=_auth(analyst)).json()["total"] == 0
    assert client.get(
        f"/api/v1/policies/{created['id']}",
        headers=_auth(analyst),
    ).status_code == 404
    assert client.patch(
        f"/api/v1/policies/{created['id']}",
        json={"title": "Not allowed"},
        headers=_auth(analyst),
    ).status_code == 404


def test_policy_crud_does_not_call_rag_service(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("policy CRUD must not call RagService")

    monkeypatch.setattr(deps, "get_rag_service", fail_if_called)
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")
    response = client.post(
        "/api/v1/policies",
        json=_payload(),
        headers=_auth(token),
    )
    assert response.status_code == 201
