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
    deps._memory_case_store = None
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
        "title": "Settlement accessibility pilot",
        "problem_statement": "Settlement growth is increasing pressure on services and land.",
        "desired_outcome": "Improve basic-service access while limiting uncontrolled expansion.",
        "territory": "Pilot district",
        "population": "Rural and peri-urban households",
        "time_horizon": "2025-2035",
        "decision_authority": "District planning authority",
        "success_criteria": ["Reduce average travel time", "Keep growth within approved boundaries"],
        "indicators": [{"name": "service_access", "unit": "minutes", "direction": "decrease"}],
        "constraints": ["Available public land", "Approved settlement boundary"],
    }


def test_case_crud_requires_authentication() -> None:
    client = _client()
    assert client.get("/api/v1/cases").status_code == 401
    assert client.post("/api/v1/cases", json=_payload()).status_code == 401


def test_case_create_update_and_source_link() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")
    headers = _auth(token)

    created = client.post("/api/v1/cases", json=_payload(), headers=headers)
    assert created.status_code == 201
    body = created.json()
    case_id = body["id"]
    assert body["revision"] == 1
    assert body["status"] == "draft"
    assert body["source_count"] == 0
    assert body["indicators"][0]["name"] == "service_access"

    linked = client.post(
        f"/api/v1/cases/{case_id}/sources",
        json={"document_id": "nludmp-2020-2050", "source_role": "policy"},
        headers=headers,
    )
    assert linked.status_code == 201
    assert linked.json()["source_role"] == "policy"

    duplicate = client.post(
        f"/api/v1/cases/{case_id}/sources",
        json={"document_id": "nludmp-2020-2050"},
        headers=headers,
    )
    assert duplicate.status_code == 409

    updated = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"status": "active", "territory": "Selected pilot district"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["status"] == "active"
    assert updated.json()["source_count"] == 1
    assert updated.json()["sources"][0]["document_id"] == "nludmp-2020-2050"

    listed = client.get("/api/v1/cases", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["cases"][0]["source_count"] == 1


def test_case_workspace_isolation() -> None:
    client = _client()
    admin = _login(client, "admin@dc-tim.ai", "admin123")
    analyst = _login(client, "analyst@dc-tim.ai", "analyst123")
    created = client.post("/api/v1/cases", json=_payload(), headers=_auth(admin)).json()

    assert client.get("/api/v1/cases", headers=_auth(analyst)).json()["total"] == 0
    assert client.get(
        f"/api/v1/cases/{created['id']}", headers=_auth(analyst)
    ).status_code == 404
    assert client.patch(
        f"/api/v1/cases/{created['id']}",
        json={"title": "Cross-workspace edit"},
        headers=_auth(analyst),
    ).status_code == 404


def test_analyst_can_create_case_but_invalid_payload_is_rejected() -> None:
    client = _client()
    token = _login(client, "analyst@dc-tim.ai", "analyst123")
    invalid = client.post(
        "/api/v1/cases",
        json={"title": "Missing required fields"},
        headers=_auth(token),
    )
    assert invalid.status_code == 422
