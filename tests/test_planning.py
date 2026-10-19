from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.auth.users import get_user_store
from app.modules.planning.schemas import ScenarioWrite


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
