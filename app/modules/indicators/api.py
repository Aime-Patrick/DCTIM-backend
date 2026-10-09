from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ...core.identity import get_workspace_id
from ...dependencies import get_case_service, get_indicator_service, require_any_permission
from ..cases.application import CaseCreateCommand, TransformationCaseService
from .application import IndicatorCreateCommand, IndicatorService, IndicatorUpdateCommand
from .domain import CaseIndicator
from .schemas import (
    IndicatorCreateRequest,
    IndicatorListResponse,
    IndicatorResponse,
    IndicatorUpdateRequest,
)

router = APIRouter(prefix="/cases/{case_id}/indicators", tags=["case-indicators"])
_READ = Depends(require_any_permission("dashboard:view", "monitoring:view", "data:view", "optimize:run"))
_WRITE = Depends(require_any_permission("policy:manage", "data:manage", "optimize:run"))


def _to_response(indicator: CaseIndicator) -> IndicatorResponse:
    return IndicatorResponse(
        id=indicator.id,
        case_id=indicator.case_id,
        workspace_id=indicator.workspace_id,
        name=indicator.name,
        definition=indicator.definition,
        unit=indicator.unit,
        direction=indicator.direction,
        baseline_value=indicator.baseline_value,
        target_value=indicator.target_value,
        current_value=indicator.current_value,
        uncertainty=indicator.uncertainty,
        quality_status=indicator.quality_status,
        source_refs=list(indicator.source_refs),
        measurement_date=indicator.measurement_date,
        metadata=indicator.metadata,
        created_at=indicator.created_at,
        updated_at=indicator.updated_at,
    )


def _ensure_case(
    workspace_id: str, case_id: str, service: TransformationCaseService
) -> None:
    if service.get_case(workspace_id, case_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transformation case not found.")


@router.get("", response_model=IndicatorListResponse, dependencies=[_READ])
def list_indicators(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    service: Annotated[IndicatorService, Depends(get_indicator_service)],
) -> IndicatorListResponse:
    _ensure_case(workspace_id, case_id, case_service)
    return IndicatorListResponse(
        indicators=[_to_response(item) for item in service.list_indicators(workspace_id, case_id)]
    )


@router.post("", response_model=IndicatorResponse, status_code=status.HTTP_201_CREATED, dependencies=[_WRITE])
def create_indicator(
    case_id: str,
    request: IndicatorCreateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    service: Annotated[IndicatorService, Depends(get_indicator_service)],
) -> IndicatorResponse:
    _ensure_case(workspace_id, case_id, case_service)
    try:
        indicator = service.create_indicator(
            workspace_id,
            case_id,
            IndicatorCreateCommand(**request.model_dump()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if indicator is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transformation case not found.")
    return _to_response(indicator)


@router.patch("/{indicator_id}", response_model=IndicatorResponse, dependencies=[_WRITE])
def update_indicator(
    case_id: str,
    indicator_id: str,
    request: IndicatorUpdateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    service: Annotated[IndicatorService, Depends(get_indicator_service)],
) -> IndicatorResponse:
    _ensure_case(workspace_id, case_id, case_service)
    try:
        indicator = service.update_indicator(
            workspace_id,
            case_id,
            indicator_id,
            IndicatorUpdateCommand(**request.model_dump()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if indicator is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Indicator not found.")
    return _to_response(indicator)
