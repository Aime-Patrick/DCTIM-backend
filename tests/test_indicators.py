from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.cases.baseline_proposals import build_baseline_proposal
from app.modules.indicators.domain import CaseIndicator
from app.modules.auth.users import get_user_store


def _client() -> TestClient:
    get_user_store.cache_clear()
    deps.get_settings.cache_clear()
    deps.get_auth_service.cache_clear()
    deps._memory_case_store = None
    deps._memory_indicator_store = None
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


def _create_case(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "title": "Settlement accessibility pilot",
            "problem_statement": "Access to services varies across settlements.",
            "desired_outcome": "Reduce access inequality.",
            "territory": "Pilot district",
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def _indicator_payload() -> dict[str, object]:
    return {
        "name": "school_travel_time",
        "definition": "Average travel time from settlement households to a primary school.",
        "unit": "minutes",
        "direction": "decrease",
        "baseline_value": 45,
        "target_value": 30,
        "current_value": 42,
        "uncertainty": 4,
        "quality_status": "trusted",
        "source_refs": ["nludmp-2020-2050"],
    }


def test_indicator_crud_and_diagnosis() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")
    headers = _auth(token)
    case_id = _create_case(client, headers)

    created = client.post(
        f"/api/v1/cases/{case_id}/indicators",
        headers=headers,
        json=_indicator_payload(),
    )
    assert created.status_code == 201
    indicator_id = created.json()["id"]
    assert created.json()["baseline_value"] == 45
    assert created.json()["quality_status"] == "trusted"

    duplicate = client.post(
        f"/api/v1/cases/{case_id}/indicators",
        headers=headers,
        json=_indicator_payload(),
    )
    assert duplicate.status_code == 409

    listed = client.get(f"/api/v1/cases/{case_id}/indicators", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()["indicators"]) == 1

    updated = client.patch(
        f"/api/v1/cases/{case_id}/indicators/{indicator_id}",
        headers=headers,
        json={"current_value": 38},
    )
    assert updated.status_code == 200
    assert updated.json()["current_value"] == 38

    diagnosis = client.get(f"/api/v1/cases/{case_id}/diagnosis", headers=headers)
    assert diagnosis.status_code == 200
    body = diagnosis.json()
    assert body["evidence_status"] == "insufficient"
    assert body["baseline_complete"] is True
    assert "No evidence sources are linked to the case." in body["evidence_gaps"]
    assert body["indicators"][0]["name"] == "school_travel_time"


def test_diagnosis_reports_partial_evidence_after_source_link() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")
    headers = _auth(token)
    case_id = _create_case(client, headers)

    linked = client.post(
        f"/api/v1/cases/{case_id}/sources",
        headers=headers,
        json={"document_id": "nludmp-2020-2050", "source_role": "policy"},
    )
    assert linked.status_code == 201
    created = client.post(
        f"/api/v1/cases/{case_id}/indicators",
        headers=headers,
        json={
            "name": "service_coverage",
            "definition": "Share of households within the agreed service threshold.",
            "unit": "percent",
            "direction": "increase",
            "baseline_value": 55,
            "target_value": 80,
            "quality_status": "partial",
            "source_refs": ["nludmp-2020-2050"],
        },
    )
    assert created.status_code == 201

    diagnosis = client.get(f"/api/v1/cases/{case_id}/diagnosis", headers=headers)
    assert diagnosis.status_code == 200
    assert diagnosis.json()["evidence_status"] == "partial"
    assert diagnosis.json()["baseline_complete"] is True


def test_indicator_workspace_isolation() -> None:
    client = _client()
    admin = _login(client, "admin@dc-tim.ai", "admin123")
    analyst = _login(client, "analyst@dc-tim.ai", "analyst123")
    case_id = _create_case(client, _auth(admin))
    created = client.post(
        f"/api/v1/cases/{case_id}/indicators",
        headers=_auth(admin),
        json=_indicator_payload(),
    )
    assert created.status_code == 201

    response = client.get(
        f"/api/v1/cases/{case_id}/indicators",
        headers=_auth(analyst),
    )
    assert response.status_code == 404


def test_prompt_driven_indicator_proposal_requires_approval() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")
    headers = _auth(token)
    case_id = _create_case(client, headers)

    proposed = client.post(
        f"/api/v1/cases/{case_id}/indicator-proposals",
        headers=headers,
        json={"prompt": "Set up the settlement indicators and identify missing data."},
    )
    assert proposed.status_code == 200
    proposal = proposed.json()
    assert proposal["generation_mode"] == "settlement_catalog_fallback"
    assert len(proposal["indicators"]) == 6
    assert client.get(f"/api/v1/cases/{case_id}/indicators", headers=headers).json()["indicators"] == []

    approved = client.post(
        f"/api/v1/cases/{case_id}/indicator-proposals/approve",
        headers=headers,
        json={
            "proposal_id": proposal["proposal_id"],
            "prompt": proposal["prompt"],
            "generation_mode": proposal["generation_mode"],
            "indicators": proposal["indicators"],
        },
    )
    assert approved.status_code == 200
    assert len(approved.json()["created"]) == 6

    listed = client.get(f"/api/v1/cases/{case_id}/indicators", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()["indicators"]) == 6


def test_baseline_proposal_maps_grounded_metric_to_existing_indicator() -> None:
    now = datetime.now(timezone.utc)
    indicator = CaseIndicator(
        id="indicator-1",
        case_id="case-1",
        workspace_id="workspace-1",
        name="Basic infrastructure coverage",
        definition="Share of households covered by basic infrastructure.",
        unit="percent",
        direction="increase",
        baseline_value=None,
        target_value=None,
        current_value=None,
        uncertainty=None,
        quality_status="unassessed",
        source_refs=(),
        measurement_date=None,
        metadata={},
        created_at=now,
        updated_at=now,
    )

    class FakeRag:
        def analyze(self, workspace_id: str, query: str, top_k: int, document_ids=None):
            return SimpleNamespace(
                analysis={
                    "metrics": [
                        {
                            "label": "Basic infrastructure coverage",
                            "baseline": 61,
                            "unit": "percent",
                            "direction": "increase",
                            "evidence_refs": ["chunk-1"],
                            "rationale": "The linked dataset states 61 percent coverage.",
                        }
                    ]
                }
            )

    package = build_baseline_proposal(
        "workspace-1",
        "case-1",
        [indicator],
        ("nludmp-source",),
        "Extract the baseline.",
        FakeRag(),  # type: ignore[arg-type]
    )
    assert package.generation_mode == "model_assisted"
    assert package.updates[0].indicator_id == "indicator-1"
    assert package.updates[0].baseline_value == 61
    assert package.updates[0].source_refs == ("chunk-1",)
