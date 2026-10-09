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
    deps._engine = None
    deps._session_factory = None
    os.environ.pop("DATABASE_URL", None)
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hash"
    os.environ["RAG_ANSWER_PROVIDER"] = "demo"
    return TestClient(create_app())


def test_case_plan_lifecycle_monitoring_and_exports() -> None:
    client = _client()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@dc-tim.ai", "password": "admin123"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    case = client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "title": "Settlement plan pilot",
            "problem_statement": "Sprawl is weakening service delivery.",
            "desired_outcome": "Improve access through planned consolidation.",
        },
    ).json()
    case_id = case["id"]
    indicator = client.post(
        f"/api/v1/cases/{case_id}/indicators",
        headers=headers,
        json={
            "name": "Service access",
            "definition": "Population within the agreed service threshold.",
            "unit": "%",
            "direction": "increase",
            "baseline_value": 20,
            "target_value": 60,
            "current_value": 30,
            "quality_status": "trusted",
            "source_refs": ["chunk-1"],
        },
    ).json()

    generated = client.post(
        f"/api/v1/cases/{case_id}/plan/generate",
        headers=headers,
        json={"decision": "Prioritise planned agglomerations", "timeline": "2026-2030"},
    )
    assert generated.status_code == 200
    assert generated.json()["status"] == "draft"
    assert generated.json()["indicators"][0]["name"] == "Service access"

    monitoring = client.get(f"/api/v1/cases/{case_id}/monitoring", headers=headers)
    assert monitoring.status_code == 200
    assert monitoring.json()["overall_status"] == "off_track"
    assert monitoring.json()["indicators"][0]["progress_percent"] == 25

    approved = client.post(f"/api/v1/cases/{case_id}/plan/approve", headers=headers)
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    pdf = client.get(f"/api/v1/cases/{case_id}/plan/export?format=pdf", headers=headers)
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")
    assert pdf.content.startswith(b"%PDF")

    markdown = client.get(f"/api/v1/cases/{case_id}/plan/export?format=markdown", headers=headers)
    assert markdown.status_code == 200
    assert "Prioritise planned agglomerations" in markdown.text

    assert indicator["id"]
