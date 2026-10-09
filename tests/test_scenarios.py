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


def test_scenario_comparison_calculates_supported_indicator_gaps() -> None:
    client = _client()
    headers = {"Authorization": f"Bearer {_login(client)}"}
    case = client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "title": "Settlement comparison pilot",
            "problem_statement": "Compare settlement growth patterns.",
            "desired_outcome": "Improve service access while controlling sprawl.",
        },
    ).json()
    case_id = case["id"]

    created = client.post(
        f"/api/v1/cases/{case_id}/indicators",
        headers=headers,
        json={
            "name": "Population within basic services",
            "definition": "Share of population within the agreed service threshold.",
            "unit": "%",
            "direction": "increase",
            "baseline_value": 10,
            "target_value": 20,
            "current_value": 15,
            "quality_status": "trusted",
            "source_refs": ["chunk-1"],
        },
    )
    assert created.status_code == 201

    response = client.post(
        f"/api/v1/cases/{case_id}/scenarios/compare",
        headers=headers,
        json={"prompt": "Compare rurban expansion with consolidated settlements."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["evidence_status"] == "ready"
    assert body["metrics"][0]["baseline_to_target_gap"] == 10
    assert body["metrics"][0]["target_gap"] == 5
    assert body["metrics"][0]["progress_percent"] == 50
    assert len(body["options"]) == 2
    assert "consolidated settlements" in body["recommendation"].lower()
