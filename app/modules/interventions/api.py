from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...core.identity import get_workspace_id
from ...dependencies import get_case_service, get_intervention_service, get_rag_service, require_any_permission
from ..cases.application import TransformationCaseService
from ..rag.application import RagService
from .application import InterventionCreateCommand, InterventionService, InterventionUpdateCommand
from .domain import CaseIntervention
from .proposals import build_intervention_proposal
from .schemas import (
    InterventionCreateRequest,
    InterventionListResponse,
    InterventionProposalApprovalRequest,
    InterventionProposalApprovalResponse,
    InterventionProposalRequest,
    InterventionProposalResponse,
    InterventionResponse,
    InterventionUpdateRequest,
)

router = APIRouter(prefix="/cases/{case_id}/interventions", tags=["case-interventions"])
_READ = Depends(require_any_permission("dashboard:view", "monitoring:view", "data:view", "optimize:run"))
_WRITE = Depends(require_any_permission("policy:manage", "data:manage", "optimize:run"))


def _to_response(intervention: CaseIntervention) -> InterventionResponse:
    return InterventionResponse(
        id=intervention.id,
        case_id=intervention.case_id,
        workspace_id=intervention.workspace_id,
        name=intervention.name,
        description=intervention.description,
        intervention_type=intervention.intervention_type,
        priority=intervention.priority,
        status=intervention.status,
        rationale=intervention.rationale,
        expected_impact=intervention.expected_impact,
        timeframe=intervention.timeframe,
        evidence_refs=list(intervention.evidence_refs),
        assumptions=list(intervention.assumptions),
        metadata=intervention.metadata,
        created_at=intervention.created_at,
        updated_at=intervention.updated_at,
    )


def _ensure_case(
    workspace_id: str, case_id: str, service: TransformationCaseService
) -> None:
    if service.get_case(workspace_id, case_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transformation case not found.")


@router.get("", response_model=InterventionListResponse, dependencies=[_READ])
def list_interventions(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    service: Annotated[InterventionService, Depends(get_intervention_service)],
) -> InterventionListResponse:
    _ensure_case(workspace_id, case_id, case_service)
    return InterventionListResponse(
        interventions=[_to_response(item) for item in service.list_interventions(workspace_id, case_id)]
    )


@router.post("", response_model=InterventionResponse, status_code=status.HTTP_201_CREATED, dependencies=[_WRITE])
def create_intervention(
    case_id: str,
    request: InterventionCreateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    service: Annotated[InterventionService, Depends(get_intervention_service)],
) -> InterventionResponse:
    _ensure_case(workspace_id, case_id, case_service)
    try:
        intervention = service.create_intervention(
            workspace_id,
            case_id,
            InterventionCreateCommand(**request.model_dump()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if intervention is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transformation case not found.")
    return _to_response(intervention)


@router.patch("/{intervention_id}", response_model=InterventionResponse, dependencies=[_WRITE])
def update_intervention(
    case_id: str,
    intervention_id: str,
    request: InterventionUpdateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    service: Annotated[InterventionService, Depends(get_intervention_service)],
) -> InterventionResponse:
    _ensure_case(workspace_id, case_id, case_service)
    try:
        intervention = service.update_intervention(
            workspace_id,
            case_id,
            intervention_id,
            InterventionUpdateCommand(**request.model_dump()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if intervention is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Intervention not found.")
    return _to_response(intervention)


@router.post("/proposals", response_model=InterventionProposalResponse, dependencies=[_WRITE])
def propose_interventions(
    case_id: str,
    request: InterventionProposalRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> InterventionProposalResponse:
    case = case_service.get_case(workspace_id, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transformation case not found.")
    package = build_intervention_proposal(case, request.prompt, rag_service)
    return InterventionProposalResponse(
        proposal_id=package.proposal_id,
        case_id=package.case_id,
        prompt=package.prompt,
        generation_mode=package.generation_mode,
        summary=package.summary,
        interventions=[
            {
                "name": item.name,
                "description": item.description,
                "intervention_type": item.intervention_type,
                "priority": item.priority,
                "rationale": item.rationale,
                "expected_impact": item.expected_impact,
                "timeframe": item.timeframe,
                "evidence_refs": list(item.evidence_refs),
                "assumptions": list(item.assumptions),
            }
            for item in package.interventions
        ],
        warnings=list(package.warnings),
    )


@router.post("/proposals/approve", response_model=InterventionProposalApprovalResponse, dependencies=[_WRITE])
def approve_interventions(
    case_id: str,
    request: InterventionProposalApprovalRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    service: Annotated[InterventionService, Depends(get_intervention_service)],
) -> InterventionProposalApprovalResponse:
    _ensure_case(workspace_id, case_id, case_service)
    existing_names = {
        item.name.casefold() for item in service.list_interventions(workspace_id, case_id)
    }
    if any(item.name.casefold() in existing_names for item in request.interventions):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="One or more proposed interventions already exist in this case.",
        )

    created: list[InterventionResponse] = []
    try:
        for item in request.interventions:
            intervention = service.create_intervention(
                workspace_id,
                case_id,
                InterventionCreateCommand(
                    name=item.name,
                    description=item.description,
                    intervention_type=item.intervention_type,
                    priority=item.priority,
                    status="approved",
                    rationale=item.rationale,
                    expected_impact=item.expected_impact,
                    timeframe=item.timeframe,
                    evidence_refs=tuple(item.evidence_refs),
                    assumptions=tuple(item.assumptions),
                    metadata={
                        "proposal_id": request.proposal_id,
                        "prompt": request.prompt,
                        "generation_mode": request.generation_mode,
                    },
                ),
            )
            if intervention is not None:
                created.append(_to_response(intervention))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return InterventionProposalApprovalResponse(proposal_id=request.proposal_id, created=created)
