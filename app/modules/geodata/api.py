from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/geodata", tags=["geodata"])

_ROAD_QUERY_URL = "https://geodata.rw/server/rest/services/basemap/Road_network/FeatureServer/0/query"
_CACHE_TTL_SECONDS = 15 * 60
_CACHE_MAX_PAGES = 32
_road_cache: OrderedDict[int, tuple[float, dict[str, Any]]] = OrderedDict()
_road_cache_lock = asyncio.Lock()


@router.get("/rwanda-roads")
async def rwanda_roads_page(
    offset: Annotated[int, Query(ge=0, le=500_000)] = 0,
) -> dict[str, Any]:
    """Proxy one bounded page of the Rwanda public road layer, avoiding browser CORS."""
    now = time.monotonic()
    async with _road_cache_lock:
        cached = _road_cache.get(offset)
        if cached and cached[0] > now:
            _road_cache.move_to_end(offset)
            return cached[1]
        if cached:
            _road_cache.pop(offset, None)

    params = {
        "where": "1=1",
        "outFields": "type,road_no,class",
        "geometry": "28.75,-2.95,31.05,-0.9",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "true",
        "resultRecordCount": "2000",
        "resultOffset": str(offset),
        "maxAllowableOffset": "0.0002",
        "geometryPrecision": "5",
        "f": "geojson",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(35.0), follow_redirects=True) as client:
            response = await client.get(_ROAD_QUERY_URL, params=params)
        if response.status_code != 200:
            logger.error("Rwanda roads provider returned HTTP %s: %s", response.status_code, response.text[:600])
            raise HTTPException(status_code=502, detail=f"Rwanda GIS road service returned HTTP {response.status_code}.")
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("features"), list):
            logger.error("Rwanda roads provider returned an invalid response: %s", response.text[:600])
            raise HTTPException(status_code=502, detail="Rwanda GIS road service returned invalid GeoJSON.")
        if result.get("error"):
            logger.error("Rwanda roads provider query error: %s", result["error"])
            raise HTTPException(status_code=502, detail="Rwanda GIS road query failed; see server logs for provider details.")
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        logger.exception("Could not fetch the Rwanda road page from geodata.rw")
        raise HTTPException(status_code=502, detail="Could not reach Rwanda GIS road service.") from exc

    async with _road_cache_lock:
        _road_cache[offset] = (time.monotonic() + _CACHE_TTL_SECONDS, result)
        _road_cache.move_to_end(offset)
        while len(_road_cache) > _CACHE_MAX_PAGES:
            _road_cache.popitem(last=False)
    return result
