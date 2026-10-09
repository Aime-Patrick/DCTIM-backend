from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from ...core.identity import get_current_user, get_workspace_id
from ...modules.auth.domain import AuthUser
from ...dependencies import get_case_service, get_indicator_service, get_intervention_service, get_rag_service, require_any_permission
from .application import CaseCreateCommand, CaseUpdateCommand, TransformationCaseService
from .diagnosis import build_diagnosis
from .domain import TransformationCase, TransformationCaseSummary
from .baseline_proposals import build_baseline_proposal
from .indicator_proposals import build_indicator_proposal
from .scenarios import build_scenario_comparison
from .plan import build_implementation_plan, build_monitoring, plan_docx, plan_markdown, plan_pdf
from ..indicators.application import IndicatorCreateCommand, IndicatorService, IndicatorUpdateCommand
from ..interventions.application import InterventionService
from ..rag.application import RagService
from .schemas import (
    CaseCreateRequest,
    CaseDetail,
    CaseListResponse,
    DiagnosisResponse,
    CaseSourceLinkRequest,
    CaseSourceResponse,
    CaseSummary,
    CaseUpdateRequest,
    BaselineProposalApprovalRequest,
    BaselineProposalApprovalResponse,
    BaselineProposalRequest,
    BaselineProposalResponse,
    IndicatorProposalApprovalRequest,
    IndicatorProposalApprovalResponse,
    IndicatorProposalRequest,
    IndicatorProposalResponse,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
    ImplementationPlanRequest,
    ImplementationPlanResponse,
    PlanGenerateRequest,
    MonitoringResponse,
)

router = APIRouter(prefix="/cases", tags=["transformation-cases"])
_CASE_READ = Depends(require_any_permission("dashboard:view", "monitoring:view", "data:view", "optimize:run"))
_CASE_WRITE = Depends(require_any_permission("policy:manage", "data:manage", "optimize:run"))


def _to_summary(summary: TransformationCaseSummary) -> CaseSummary:
    return CaseSummary(
        id=summary.id,
        title=summary.title,
        territory=summary.territory,
        status=summary.status,
        revision=summary.revision,
        source_count=summary.source_count,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


def _to_detail(case: TransformationCase) -> CaseDetail:
    return CaseDetail(
        id=case.id,
        workspace_id=case.workspace_id,
        title=case.title,
        territory=case.territory,
        status=case.status,
        revision=case.revision,
        source_count=len(case.sources),
        created_at=case.created_at,
        updated_at=case.updated_at,
        problem_statement=case.problem_statement,
        desired_outcome=case.desired_outcome,
        population=case.population,
        time_horizon=case.time_horizon,
        decision_authority=case.decision_authority,
        success_criteria=list(case.success_criteria),
        indicators=list(case.indicators),
        constraints=list(case.constraints),
        metadata=case.metadata,
        review_date=case.review_date,
        sources=[
            CaseSourceResponse(
                id=source.id,
                case_id=source.case_id,
                document_id=source.document_id,
                source_role=source.source_role,
                linked_at=source.linked_at,
            )
            for source in case.sources
        ],
    )


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transformation case not found.")


@router.post("", response_model=CaseDetail, status_code=status.HTTP_201_CREATED, dependencies=[_CASE_WRITE])
def create_case(
    request: CaseCreateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
) -> CaseDetail:
    try:
        case = service.create_case(
            workspace_id,
            CaseCreateCommand(
                title=request.title,
                problem_statement=request.problem_statement,
                desired_outcome=request.desired_outcome,
                territory=request.territory,
                population=request.population,
                time_horizon=request.time_horizon,
                decision_authority=request.decision_authority,
                status=request.status,
                success_criteria=tuple(request.success_criteria),
                indicators=tuple(request.indicators),
                constraints=tuple(request.constraints),
                metadata=request.metadata,
                review_date=request.review_date,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _to_detail(case)


@router.get("", response_model=CaseListResponse, dependencies=[_CASE_READ])
def list_cases(
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CaseListResponse:
    cases, total = service.list_cases(workspace_id, limit=limit, offset=offset)
    return CaseListResponse(cases=[_to_summary(case) for case in cases], total=total)


@router.get("/{case_id}", response_model=CaseDetail, dependencies=[_CASE_READ])
def get_case(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
) -> CaseDetail:
    case = service.get_case(workspace_id, case_id)
    if case is None:
        raise _not_found()
    return _to_detail(case)


@router.patch("/{case_id}", response_model=CaseDetail, dependencies=[_CASE_WRITE])
def update_case(
    case_id: str,
    request: CaseUpdateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
) -> CaseDetail:
    try:
        case = service.update_case(
            workspace_id,
            case_id,
            CaseUpdateCommand(**request.model_dump()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if case is None:
        raise _not_found()
    return _to_detail(case)


@router.post(
    "/{case_id}/sources",
    response_model=CaseSourceResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_CASE_WRITE],
)
def link_source(
    case_id: str,
    request: CaseSourceLinkRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
) -> CaseSourceResponse:
    try:
        source = service.link_source(
            workspace_id, case_id, request.document_id, request.source_role
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if source is None:
        raise _not_found()
    return CaseSourceResponse(
        id=source.id,
        case_id=source.case_id,
        document_id=source.document_id,
        source_role=source.source_role,
        linked_at=source.linked_at,
    )


@router.post(
    "/{case_id}/indicator-proposals",
    response_model=IndicatorProposalResponse,
    dependencies=[_CASE_WRITE],
)
def propose_indicators(
    case_id: str,
    request: IndicatorProposalRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> IndicatorProposalResponse:
    case = case_service.get_case(workspace_id, case_id)
    if case is None:
        raise _not_found()
    package = build_indicator_proposal(case, request.prompt, rag_service)
    return IndicatorProposalResponse(
        proposal_id=package.proposal_id,
        case_id=package.case_id,
        prompt=package.prompt,
        generation_mode=package.generation_mode,
        summary=package.summary,
        indicators=[
            {
                "name": item.name,
                "definition": item.definition,
                "unit": item.unit,
                "direction": item.direction,
                "baseline_value": item.baseline_value,
                "target_value": item.target_value,
                "source_refs": list(item.source_refs),
                "rationale": item.rationale,
            }
            for item in package.indicators
        ],
        evidence_source_ids=list(package.evidence_source_ids),
        warnings=list(package.warnings),
    )


@router.post(
    "/{case_id}/indicator-proposals/approve",
    response_model=IndicatorProposalApprovalResponse,
    dependencies=[_CASE_WRITE],
)
def approve_indicators(
    case_id: str,
    request: IndicatorProposalApprovalRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    indicator_service: Annotated[IndicatorService, Depends(get_indicator_service)],
) -> IndicatorProposalApprovalResponse:
    case = case_service.get_case(workspace_id, case_id)
    if case is None:
        raise _not_found()
    existing_names = {
        item.name.casefold()
        for item in indicator_service.list_indicators(workspace_id, case_id)
    }
    if any(item.name.casefold() in existing_names for item in request.indicators):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="One or more proposed indicators already exist in this case.",
        )

    created = []
    try:
        for item in request.indicators:
            indicator = indicator_service.create_indicator(
                workspace_id,
                case_id,
                IndicatorCreateCommand(
                    name=item.name,
                    definition=item.definition,
                    unit=item.unit,
                    direction=item.direction,
                    baseline_value=item.baseline_value,
                    target_value=item.target_value,
                    quality_status="partial" if item.source_refs else "unassessed",
                    source_refs=tuple(item.source_refs),
                    metadata={
                        "proposal_id": request.proposal_id,
                        "prompt": request.prompt,
                        "generation_mode": request.generation_mode,
                        "proposal_rationale": item.rationale,
                    },
                ),
            )
            if indicator is not None:
                created.append(
                    {
                        "id": indicator.id,
                        "case_id": indicator.case_id,
                        "workspace_id": indicator.workspace_id,
                        "name": indicator.name,
                        "definition": indicator.definition,
                        "unit": indicator.unit,
                        "direction": indicator.direction,
                        "baseline_value": indicator.baseline_value,
                        "target_value": indicator.target_value,
                        "current_value": indicator.current_value,
                        "uncertainty": indicator.uncertainty,
                        "quality_status": indicator.quality_status,
                        "source_refs": list(indicator.source_refs),
                        "measurement_date": indicator.measurement_date,
                        "metadata": indicator.metadata,
                        "created_at": indicator.created_at,
                        "updated_at": indicator.updated_at,
                    }
                )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return IndicatorProposalApprovalResponse(
        proposal_id=request.proposal_id,
        created=created,
    )


@router.post(
    "/{case_id}/baseline-proposals",
    response_model=BaselineProposalResponse,
    dependencies=[_CASE_WRITE],
)
def propose_baselines(
    case_id: str,
    request: BaselineProposalRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    indicator_service: Annotated[IndicatorService, Depends(get_indicator_service)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> BaselineProposalResponse:
    case = case_service.get_case(workspace_id, case_id)
    if case is None:
        raise _not_found()
    package = build_baseline_proposal(
        workspace_id,
        case.id,
        indicator_service.list_indicators(workspace_id, case_id),
        tuple(source.document_id for source in case.sources),
        request.prompt,
        rag_service,
    )
    return BaselineProposalResponse(
        proposal_id=package.proposal_id,
        case_id=package.case_id,
        prompt=package.prompt,
        generation_mode=package.generation_mode,
        summary=package.summary,
        updates=[
            {
                "indicator_id": item.indicator_id,
                "indicator_name": item.indicator_name,
                "baseline_value": item.baseline_value,
                "current_value": item.current_value,
                "uncertainty": item.uncertainty,
                "quality_status": item.quality_status,
                "source_refs": list(item.source_refs),
                "rationale": item.rationale,
            }
            for item in package.updates
        ],
        warnings=list(package.warnings),
    )


@router.post(
    "/{case_id}/baseline-proposals/approve",
    response_model=BaselineProposalApprovalResponse,
    dependencies=[_CASE_WRITE],
)
def approve_baselines(
    case_id: str,
    request: BaselineProposalApprovalRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    indicator_service: Annotated[IndicatorService, Depends(get_indicator_service)],
) -> BaselineProposalApprovalResponse:
    if case_service.get_case(workspace_id, case_id) is None:
        raise _not_found()
    if len({item.indicator_id for item in request.updates}) != len(request.updates):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A baseline proposal cannot update the same indicator twice.",
        )

    updated = []
    try:
        for item in request.updates:
            existing = indicator_service.get_indicator(workspace_id, case_id, item.indicator_id)
            if existing is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Indicator not found.")
            metadata = {
                **existing.metadata,
                "baseline_proposal_id": request.proposal_id,
                "baseline_prompt": request.prompt,
                "baseline_generation_mode": request.generation_mode,
                "baseline_rationale": item.rationale,
            }
            indicator = indicator_service.update_indicator(
                workspace_id,
                case_id,
                item.indicator_id,
                IndicatorUpdateCommand(
                    baseline_value=item.baseline_value,
                    current_value=item.current_value,
                    uncertainty=item.uncertainty,
                    quality_status=item.quality_status,
                    source_refs=tuple(item.source_refs),
                    metadata=metadata,
                ),
            )
            if indicator is not None:
                updated.append(
                    {
                        "id": indicator.id,
                        "case_id": indicator.case_id,
                        "workspace_id": indicator.workspace_id,
                        "name": indicator.name,
                        "definition": indicator.definition,
                        "unit": indicator.unit,
                        "direction": indicator.direction,
                        "baseline_value": indicator.baseline_value,
                        "target_value": indicator.target_value,
                        "current_value": indicator.current_value,
                        "uncertainty": indicator.uncertainty,
                        "quality_status": indicator.quality_status,
                        "source_refs": list(indicator.source_refs),
                        "measurement_date": indicator.measurement_date,
                        "metadata": indicator.metadata,
                        "created_at": indicator.created_at,
                        "updated_at": indicator.updated_at,
                    }
                )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return BaselineProposalApprovalResponse(
        proposal_id=request.proposal_id,
        updated=updated,
    )


@router.get("/{case_id}/diagnosis", response_model=DiagnosisResponse, dependencies=[_CASE_READ])
def get_diagnosis(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    indicator_service: Annotated[object, Depends(get_indicator_service)],
) -> DiagnosisResponse:
    case = case_service.get_case(workspace_id, case_id)
    if case is None:
        raise _not_found()
    package = build_diagnosis(
        case,
        indicator_service.list_indicators(workspace_id, case_id),
    )
    return DiagnosisResponse(
        case_id=package.case_id,
        generated_at=package.generated_at,
        evidence_status=package.evidence_status,
        summary=package.summary,
        baseline_complete=package.baseline_complete,
        indicators=[
            {
                "id": indicator.id,
                "name": indicator.name,
                "unit": indicator.unit,
                "direction": indicator.direction,
                "baseline_value": indicator.baseline_value,
                "target_value": indicator.target_value,
                "current_value": indicator.current_value,
                "uncertainty": indicator.uncertainty,
                "quality_status": indicator.quality_status,
                "source_refs": list(indicator.source_refs),
            }
            for indicator in package.indicators
        ],
        evidence_source_ids=list(package.evidence_source_ids),
        evidence_gaps=list(package.evidence_gaps),
        assumptions=list(package.assumptions),
        next_steps=list(package.next_steps),
    )


@router.post("/{case_id}/scenarios/compare", response_model=ScenarioComparisonResponse, dependencies=[_CASE_READ])
def compare_scenarios(
    case_id: str,
    request: ScenarioComparisonRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    indicator_service: Annotated[IndicatorService, Depends(get_indicator_service)],
    intervention_service: Annotated[InterventionService, Depends(get_intervention_service)],
) -> ScenarioComparisonResponse:
    case = case_service.get_case(workspace_id, case_id)
    if case is None:
        raise _not_found()
    package = build_scenario_comparison(
        case,
        indicator_service.list_indicators(workspace_id, case_id),
        intervention_service.list_interventions(workspace_id, case_id),
        request.prompt,
    )
    return ScenarioComparisonResponse(
        case_id=package.case_id,
        prompt=package.prompt,
        generated_at=package.generated_at,
        evidence_status=package.evidence_status,
        recommendation=package.recommendation,
        metrics=[
            {
                "indicator_id": item.indicator_id,
                "name": item.name,
                "unit": item.unit,
                "direction": item.direction,
                "baseline_value": item.baseline_value,
                "target_value": item.target_value,
                "current_value": item.current_value,
                "target_gap": item.target_gap,
                "baseline_to_target_gap": item.baseline_to_target_gap,
                "progress_percent": item.progress_percent,
                "equation": item.equation,
                "status": item.status,
                "source_refs": list(item.source_refs),
            }
            for item in package.metrics
        ],
        options=[
            {
                "key": item.key,
                "name": item.name,
                "service_access": item.service_access,
                "sprawl": item.sprawl,
                "environmental_risk": item.environmental_risk,
                "quantification_status": item.quantification_status,
                "rationale": item.rationale,
            }
            for item in package.options
        ],
        equations=list(package.equations),
        assumptions=list(package.assumptions),
        evidence_gaps=list(package.evidence_gaps),
        approved_intervention_count=package.approved_intervention_count,
    )


def _plan_response(case_id: str, plan: dict) -> ImplementationPlanResponse:
    return ImplementationPlanResponse(case_id=case_id, **{key: value for key, value in plan.items() if key != "case_id"})


def _case_with_plan(
    workspace_id: str,
    case_id: str,
    service: TransformationCaseService,
) -> tuple[TransformationCase, dict] | None:
    case = service.get_case(workspace_id, case_id)
    if case is None:
        return None
    plan = case.metadata.get("implementation_plan")
    if not isinstance(plan, dict):
        return case, {}
    return case, plan


def _save_plan(
    workspace_id: str,
    case: TransformationCase,
    service: TransformationCaseService,
    plan: dict,
    *,
    status_value: str | None = None,
) -> dict:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    saved = {
        **plan,
        "case_id": case.id,
        "version": max(1, int(plan.get("version", 1))),
        "updated_at": now,
    }
    if "created_at" not in saved:
        saved["created_at"] = now
    if status_value is not None:
        saved["status"] = status_value
    metadata = {**case.metadata, "implementation_plan": saved}
    updated = service.update_case(
        workspace_id,
        case.id,
        CaseUpdateCommand(metadata=metadata),
    )
    if updated is None:
        raise _not_found()
    return saved


@router.get(
    "/{case_id}/plan",
    response_model=ImplementationPlanResponse,
    dependencies=[_CASE_READ],
)
def get_plan(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
) -> ImplementationPlanResponse:
    result = _case_with_plan(workspace_id, case_id, service)
    if result is None:
        raise _not_found()
    case, plan = result
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Implementation plan has not been created.")
    return _plan_response(case.id, plan)


@router.post(
    "/{case_id}/plan/generate",
    response_model=ImplementationPlanResponse,
    dependencies=[_CASE_WRITE],
)
def generate_plan(
    case_id: str,
    request: PlanGenerateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    indicator_service: Annotated[object, Depends(get_indicator_service)],
    intervention_service: Annotated[InterventionService, Depends(get_intervention_service)],
) -> ImplementationPlanResponse:
    case = case_service.get_case(workspace_id, case_id)
    if case is None:
        raise _not_found()
    plan = build_implementation_plan(
        case,
        indicator_service.list_indicators(workspace_id, case_id),
        intervention_service.list_interventions(workspace_id, case_id),
        decision=request.decision,
        timeline=request.timeline,
        notes=request.notes,
    )
    return _plan_response(case.id, _save_plan(workspace_id, case, case_service, plan))


@router.put(
    "/{case_id}/plan",
    response_model=ImplementationPlanResponse,
    dependencies=[_CASE_WRITE],
)
def update_plan(
    case_id: str,
    request: ImplementationPlanRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
) -> ImplementationPlanResponse:
    result = _case_with_plan(workspace_id, case_id, service)
    if result is None:
        raise _not_found()
    case, existing = result
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    payload = request.model_dump(mode="json")
    plan = {
        **payload,
        "status": existing.get("status", "draft"),
        "version": max(1, int(existing.get("version", 0)) + 1),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "approved_at": existing.get("approved_at"),
        "approved_by": existing.get("approved_by"),
    }
    return _plan_response(case.id, _save_plan(workspace_id, case, service, plan))


@router.post(
    "/{case_id}/plan/approve",
    response_model=ImplementationPlanResponse,
    dependencies=[_CASE_WRITE],
)
def approve_plan(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
) -> ImplementationPlanResponse:
    result = _case_with_plan(workspace_id, case_id, service)
    if result is None:
        raise _not_found()
    case, plan = result
    if not plan:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Generate an implementation plan before approving it.")
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    approved = {**plan, "status": "approved", "approved_at": now, "approved_by": user.email, "updated_at": now}
    metadata = {**case.metadata, "implementation_plan": approved}
    updated = service.update_case(
        workspace_id,
        case_id,
        CaseUpdateCommand(metadata=metadata, status="active"),
    )
    if updated is None:
        raise _not_found()
    return _plan_response(case_id, approved)


@router.get(
    "/{case_id}/plan/export",
    dependencies=[_CASE_READ],
)
def export_plan(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[TransformationCaseService, Depends(get_case_service)],
    format: Annotated[Literal["pdf", "docx", "markdown", "json"], Query()] = "pdf",
) -> Response:
    result = _case_with_plan(workspace_id, case_id, service)
    if result is None:
        raise _not_found()
    case, plan = result
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Implementation plan has not been created.")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in case.title).strip("_") or "settlement_plan"
    if format == "markdown":
        return Response(plan_markdown(case, plan), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="{safe_name}.md"'})
    if format == "docx":
        return Response(plan_docx(case, plan), media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": f'attachment; filename="{safe_name}.docx"'})
    if format == "json":
        return Response(json.dumps(plan, indent=2, ensure_ascii=False), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{safe_name}.json"'})
    return Response(plan_pdf(case, plan), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'})


@router.get(
    "/{case_id}/monitoring",
    response_model=MonitoringResponse,
    dependencies=[_CASE_READ],
)
def get_monitoring(
    case_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    case_service: Annotated[TransformationCaseService, Depends(get_case_service)],
    indicator_service: Annotated[object, Depends(get_indicator_service)],
) -> MonitoringResponse:
    if case_service.get_case(workspace_id, case_id) is None:
        raise _not_found()
    monitoring = build_monitoring(indicator_service.list_indicators(workspace_id, case_id))
    return MonitoringResponse(case_id=case_id, **monitoring)
