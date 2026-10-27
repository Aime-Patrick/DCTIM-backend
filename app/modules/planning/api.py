from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...dependencies import _get_db_session, require_any_permission, require_permission
from ..auth.domain import AuthUser
from ..policies.infrastructure.models import PolicyArtifact as PolicyArtifactRow
from .models import PlanningScenarioRow
from .application import merge_policy_features, policy_scenario_id
from .schemas import PolicyProposalSyncRequest, ScenarioResponse, ScenarioWrite

router = APIRouter(prefix="/planning/scenarios", tags=["planning"])
Planner = Annotated[AuthUser, Depends(require_permission("planning:manage"))]
ScenarioReader = Annotated[AuthUser, Depends(require_any_permission("planning:manage", "monitoring:view"))]
DbSession = Annotated[object, Depends(_get_db_session)]


def _db(session: object) -> Session:
    if session is None:
        raise HTTPException(status_code=503, detail="Planning scenarios require the configured database.")
    return session  # type: ignore[return-value]


def _response(row: PlanningScenarioRow) -> ScenarioResponse:
    return ScenarioResponse(id=row.id, name=row.name, year=row.year, policy_id=row.policy_id, features=row.features, created_at=row.created_at, updated_at=row.updated_at)


@router.get("", response_model=list[ScenarioResponse])
def list_scenarios(
    user: ScenarioReader,
    session: DbSession,
    policy_id: str | None = Query(default=None, max_length=64),
    include_unlinked: bool = Query(default=False),
) -> list[ScenarioResponse]:
    query = select(PlanningScenarioRow).where(PlanningScenarioRow.workspace_id == user.workspace_id)
    if policy_id is not None:
        if include_unlinked:
            if "planning:manage" not in user.permissions:
                raise HTTPException(status_code=403, detail="Only planners can include unlinked scenarios.")
            query = query.where((PlanningScenarioRow.policy_id == policy_id) | (PlanningScenarioRow.policy_id.is_(None)))
        else:
            query = query.where(PlanningScenarioRow.policy_id == policy_id)
    rows = _db(session).scalars(query.order_by(PlanningScenarioRow.updated_at.desc())).all()
    return [_response(row) for row in rows]


@router.post("", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED)
def create_scenario(request: ScenarioWrite, user: Planner, session: DbSession) -> ScenarioResponse:
    db = _db(session)
    row = PlanningScenarioRow(id=uuid4().hex, workspace_id=user.workspace_id, created_by=user.id, policy_id=request.policy_id, name=request.name, year=request.year, features=request.features)
    db.add(row)
    db.flush()
    return _response(row)


@router.post("/policy-proposals/sync", response_model=ScenarioResponse)
def sync_policy_proposals(
    request: PolicyProposalSyncRequest,
    user: Planner,
    session: DbSession,
) -> ScenarioResponse:
    """Persist AI-generated policy features idempotently in a dedicated plan.

    Human-created features and user-edited AI features are retained on subsequent
    synchronizations. Workspace and policy ownership are checked server-side.
    """
    db = _db(session)
    policy = db.scalar(select(PolicyArtifactRow).where(
        PolicyArtifactRow.id == request.policy_id,
        PolicyArtifactRow.workspace_id == user.workspace_id,
    ))
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy artifact not found in this workspace.")

    for feature in request.features["features"]:
        if feature["properties"].get("policy_id") != request.policy_id:
            raise HTTPException(status_code=422, detail="Each generated feature must reference the requested policy.")

    scenario_id = policy_scenario_id(user.workspace_id, request.policy_id)
    row = db.scalar(select(PlanningScenarioRow).where(
        PlanningScenarioRow.id == scenario_id,
        PlanningScenarioRow.workspace_id == user.workspace_id,
    ))
    if row is None:
        row = PlanningScenarioRow(
            id=scenario_id,
            workspace_id=user.workspace_id,
            created_by=user.id,
            policy_id=request.policy_id,
            name=request.name,
            year=request.year,
            features={"type": "FeatureCollection", "features": []},
        )
        db.add(row)
        db.flush()
    elif row.policy_id != request.policy_id:
        raise HTTPException(status_code=409, detail="Policy plan identifier collision.")

    existing = list((row.features or {}).get("features", []))
    merged = merge_policy_features(existing, request.features["features"])

    row.name = request.name
    row.year = request.year
    row.features = {"type": "FeatureCollection", "features": merged}
    db.flush()
    return _response(row)


@router.put("/{scenario_id}", response_model=ScenarioResponse)
def update_scenario(scenario_id: str, request: ScenarioWrite, user: Planner, session: DbSession) -> ScenarioResponse:
    db = _db(session)
    row = db.scalar(select(PlanningScenarioRow).where(PlanningScenarioRow.id == scenario_id, PlanningScenarioRow.workspace_id == user.workspace_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Planning scenario not found.")
    row.name, row.year, row.policy_id, row.features = request.name, request.year, request.policy_id, request.features
    db.flush()
    return _response(row)


@router.delete("/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scenario(scenario_id: str, user: Planner, session: DbSession) -> Response:
    db = _db(session)
    row = db.scalar(select(PlanningScenarioRow).where(PlanningScenarioRow.id == scenario_id, PlanningScenarioRow.workspace_id == user.workspace_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Planning scenario not found.")
    db.delete(row)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
