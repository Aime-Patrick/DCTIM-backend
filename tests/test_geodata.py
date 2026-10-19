from __future__ import annotations

import os

import httpx
from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.geodata import api as geodata_api


def test_rwanda_roads_proxy_returns_geojson_and_caches_page(monkeypatch) -> None:
    calls = 0

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params):
            nonlocal calls
            calls += 1
            assert url == geodata_api._ROAD_QUERY_URL
            assert params["resultOffset"] == "0"
            return httpx.Response(200, json={"type": "FeatureCollection", "features": []})

    deps.get_settings.cache_clear()
    deps._engine = None
    deps._session_factory = None
    os.environ.pop("DATABASE_URL", None)
    geodata_api._road_cache.clear()
    monkeypatch.setattr(geodata_api.httpx, "AsyncClient", MockClient)
    client = TestClient(create_app())

    first = client.get("/api/v1/geodata/rwanda-roads?offset=0")
    second = client.get("/api/v1/geodata/rwanda-roads?offset=0")

    assert first.status_code == 200
    assert first.json()["type"] == "FeatureCollection"
    assert second.status_code == 200
    assert calls == 1
