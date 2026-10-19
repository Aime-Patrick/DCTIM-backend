from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


def validate_feature_collection(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("type") != "FeatureCollection" or not isinstance(value.get("features"), list):
        raise ValueError("features must be a GeoJSON FeatureCollection")
    if len(value["features"]) > 5000:
        raise ValueError("A scenario can contain at most 5000 features")

    def coordinates(node: Any) -> None:
        if not isinstance(node, list) or len(node) == 0:
            raise ValueError("GeoJSON coordinates must be non-empty arrays")
        if isinstance(node[0], (int, float)):
            if len(node) < 2 or not all(isinstance(v, (int, float)) for v in node[:2]):
                raise ValueError("Coordinates must contain numeric longitude and latitude")
            lon, lat = node[:2]
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("Coordinates must use WGS84 longitude and latitude")
            return
        for child in node:
            coordinates(child)

    feature_ids: set[str] = set()
    for feature in value["features"]:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("Each scenario item must be a GeoJSON Feature")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in {"Point", "LineString", "Polygon"}:
            raise ValueError("Supported GeoJSON geometries are Point, LineString, and Polygon")
        raw_coordinates = geometry.get("coordinates")
        coordinates(raw_coordinates)
        if geometry["type"] == "Point" and (not isinstance(raw_coordinates, list) or len(raw_coordinates) < 2 or isinstance(raw_coordinates[0], list)):
            raise ValueError("Point geometry requires one longitude/latitude coordinate")
        if geometry["type"] == "LineString" and (not isinstance(raw_coordinates, list) or len(raw_coordinates) < 2):
            raise ValueError("LineString geometry requires at least two coordinates")
        if geometry["type"] == "Polygon":
            if not isinstance(raw_coordinates, list) or not raw_coordinates or any(not isinstance(ring, list) or len(ring) < 4 or ring[0][:2] != ring[-1][:2] for ring in raw_coordinates):
                raise ValueError("Polygon rings must have at least four coordinates and be closed")
            if _self_intersects(raw_coordinates[0]):
                raise ValueError("Polygon outer ring must not self-intersect")
        if not isinstance(feature.get("id"), str) or not feature["id"].strip():
            raise ValueError("Every GeoJSON Feature requires a non-empty id")
        if feature["id"] in feature_ids:
            raise ValueError("GeoJSON Feature ids must be unique within a scenario")
        feature_ids.add(feature["id"])
        if not isinstance(feature.get("properties", {}), dict):
            raise ValueError("Feature properties must be an object")
    return value


def _self_intersects(ring: list[list[float]]) -> bool:
    """Test segment crossings in local lon/lat coordinates (planning-scale polygons)."""
    segments = list(zip(ring, ring[1:]))
    for i, (a, b) in enumerate(segments):
        for j, (c, d) in enumerate(segments):
            if abs(i - j) <= 1 or (i == 0 and j == len(segments) - 1) or (j == 0 and i == len(segments) - 1):
                continue
            orient = lambda p, q, r: (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                return True
    return False


class ScenarioWrite(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    year: int = Field(ge=2024, le=2200)
    policy_id: str | None = Field(default=None, max_length=64)
    features: dict[str, Any]

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("Scenario name cannot be blank")
        return result

    @field_validator("features")
    @classmethod
    def valid_geojson(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_feature_collection(value)


class ScenarioResponse(BaseModel):
    id: str
    name: str
    year: int
    policy_id: str | None = None
    features: dict[str, Any]
    created_at: datetime
    updated_at: datetime
