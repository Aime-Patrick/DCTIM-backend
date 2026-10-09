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
    deps._memory_indicator_store = None
    deps._memory_intervention_store = None
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
    assert response.status_code == 200
    return response.json()["access_token"]


def test_intervention_proposal_requires_approval() -> None:
    client = _client()
    headers = {"Authorization": f"Bearer {_login(client)}"}
    created_case = client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "title": "Settlement accessibility pilot",
            "problem_statement": "Settlement growth is increasing service-access inequality.",
            "desired_outcome": "Improve access while controlling unplanned expansion.",
            "territory": "Pilot district",
        },
    )
    assert created_case.status_code == 201
    case_id = created_case.json()["id"]

    proposed = client.post(
        f"/api/v1/cases/{case_id}/interventions/proposals",
        headers=headers,
        json={"prompt": "Propose interventions for service access, boundaries, and environmental risk."},
    )
    assert proposed.status_code == 200
    proposal = proposed.json()
    assert proposal["generation_mode"] == "settlement_catalog_fallback"
    assert len(proposal["interventions"]) == 5
    assert client.get(f"/api/v1/cases/{case_id}/interventions", headers=headers).json()["interventions"] == []

    approved = client.post(
        f"/api/v1/cases/{case_id}/interventions/proposals/approve",
        headers=headers,
        json={
            "proposal_id": proposal["proposal_id"],
            "prompt": proposal["prompt"],
            "generation_mode": proposal["generation_mode"],
            "interventions": proposal["interventions"],
        },
    )
    assert approved.status_code == 200
    assert len(approved.json()["created"]) == 5
    assert all(item["status"] == "approved" for item in approved.json()["created"])


def test_duplicate_intervention_is_rejected() -> None:
    client = _client()
    headers = {"Authorization": f"Bearer {_login(client)}"}
    case = client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "title": "Settlement pilot",
            "problem_statement": "Service access varies.",
            "desired_outcome": "Improve access.",
        },
    ).json()
    case_id = case["id"]
    payload = {
        "name": "Boundary control",
        "description": "Control expansion.",
        "intervention_type": "boundary_control",
    }
    assert client.post(f"/api/v1/cases/{case_id}/interventions", headers=headers, json=payload).status_code == 201
    duplicate = client.post(f"/api/v1/cases/{case_id}/interventions", headers=headers, json=payload)
    assert duplicate.status_code == 409
