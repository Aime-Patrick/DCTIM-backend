from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.auth.users import get_user_store
from app.modules.planning.application import merge_policy_features, policy_scenario_id
from app.modules.planning.schemas import PolicyProposalSyncRequest, ScenarioWrite


def _client() -> TestClient:
    get_user_store.cache_clear()
    deps.get_settings.cache_clear()
    deps.get_auth_service.cache_clear()
    deps._engine = None
    deps._session_factory = None
    os.environ.pop("DATABASE_URL", None)
    return TestClient(create_app())


def _token(client: TestClient, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_monitoring_users_can_read_plans_but_need_database() -> None:
    client = _client()
    token = _token(client, "analyst@dc-tim.ai", "analyst123")
    response = client.get("/api/v1/planning/scenarios", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 503
    assert "configured database" in response.json()["detail"]


def test_planning_api_requires_database_for_persistence() -> None:
    client = _client()
    token = _token(client, "admin@dc-tim.ai", "admin123")
    response = client.get("/api/v1/planning/scenarios", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 503
    assert "configured database" in response.json()["detail"]


def test_scenario_rejects_out_of_range_wgs84_coordinates() -> None:
    with pytest.raises(ValueError, match="WGS84"):
        ScenarioWrite.model_validate({
            "name": "Invalid",
            "year": 2035,
            "features": {"type": "FeatureCollection", "features": [{
                "type": "Feature", "geometry": {"type": "Point", "coordinates": [300, -2]}, "properties": {},
            }]},
        })


def test_planning_scenario_accepts_multiline_and_multipolygon_geometry() -> None:
    payload = {
        "name": "Geometry formats",
        "year": 2035,
        "features": {"type": "FeatureCollection", "features": [
            {
                "type": "Feature", "id": "road-multi",
                "geometry": {"type": "MultiLineString", "coordinates": [
                    [[30.0, -2.0], [30.1, -2.1]],
                    [[30.2, -2.0], [30.3, -2.1]],
                ]},
                "properties": {"layer": "proposed-road"},
            },
            {
                "type": "Feature", "id": "buildings-multi",
                "geometry": {"type": "MultiPolygon", "coordinates": [
                    [[[30.0, -2.0], [30.01, -2.0], [30.01, -2.01], [30.0, -2.0]]],
                    [[[30.1, -2.0], [30.11, -2.0], [30.11, -2.01], [30.1, -2.0]]],
                ]},
                "properties": {"layer": "proposed-building", "height": 12},
            },
        ]},
    }
    assert len(ScenarioWrite.model_validate(payload).features["features"]) == 2


def test_scenario_rejects_multiline_component_with_one_vertex() -> None:
    with pytest.raises(ValueError, match="MultiLineString"):
        ScenarioWrite.model_validate({
            "name": "Invalid route",
            "year": 2035,
            "features": {"type": "FeatureCollection", "features": [{
                "type": "Feature", "id": "road-invalid",
                "geometry": {"type": "MultiLineString", "coordinates": [[[30.0, -2.0]]]},
                "properties": {},
            }]},
        })


def test_policy_sync_requires_provenance_and_policy_id() -> None:
    payload = {
        "policy_id": "policy-1",
        "name": "AI Policy Plan",
        "year": 2035,
        "features": {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "id": "stable-proposal-1",
            "geometry": {"type": "Point", "coordinates": [30.1, -1.9]},
            "properties": {
                "policy_id": "policy-1",
                "ai_proposal_id": "proposal-1",
                "proposal_source": "AI/RAG policy suggestion",
            },
        }]},
    }
    assert PolicyProposalSyncRequest.model_validate(payload).policy_id == "policy-1"
    payload["features"]["features"][0]["properties"]["proposal_source"] = "surveyed asset"
    with pytest.raises(ValueError, match="AI/RAG"):
        PolicyProposalSyncRequest.model_validate(payload)


def test_policy_proposal_merge_is_idempotent_and_keeps_user_edits() -> None:
    def feature(name: str, *, edited: bool = False, proposal_id: str = "proposal-1") -> dict:
        return {
            "type": "Feature",
            "id": f"feature-{proposal_id}",
            "geometry": {"type": "Point", "coordinates": [30.1, -1.9]},
            "properties": {"name": name, "ai_proposal_id": proposal_id, "user_modified": edited},
        }

    original_manual = feature("planner created", proposal_id="manual")
    existing = [original_manual, feature("edited name", edited=True)]
    incoming = [feature("new AI label"), feature("another proposal", proposal_id="proposal-2")]
    first = merge_policy_features(existing, incoming)
    second = merge_policy_features(first, incoming)
    assert len(second) == 3
    assert second[0] == original_manual
    assert second[1]["properties"]["name"] == "edited name"
    assert second[2]["properties"]["name"] == "another proposal"


def test_policy_plan_ids_are_stable_and_workspace_scoped() -> None:
    assert policy_scenario_id("workspace-a", "policy-a") == policy_scenario_id("workspace-a", "policy-a")
    assert policy_scenario_id("workspace-a", "policy-a") != policy_scenario_id("workspace-b", "policy-a")
