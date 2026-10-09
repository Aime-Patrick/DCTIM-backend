from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from ...core.identity import get_workspace_id
from ...dependencies import get_policy_service, require_any_permission, require_permission
from .application import PolicyCreateCommand, PolicyService, PolicyUpdateCommand
from .domain import PolicyArtifact, PolicySummary as DomainPolicySummary
from .export import policy_docx, policy_markdown, policy_pdf
from .schemas import (
    PolicyCreateRequest,
    PolicyDetail,
    PolicyListResponse,
    PolicySummary,
    PolicyUpdateRequest,
)

router = APIRouter(prefix="/policies", tags=["policies"])


def _to_detail(artifact: PolicyArtifact) -> PolicyDetail:
    return PolicyDetail(
        id=artifact.id,
        title=artifact.title,
        content=artifact.content,
        category=artifact.category,
        status=artifact.status,
        analysis_trace_id=artifact.analysis_trace_id,
        analysis_provider=artifact.analysis_provider,
        has_analysis=artifact.analysis is not None,
        analysis=artifact.analysis,
        revision=artifact.revision,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


def _to_summary(summary: DomainPolicySummary) -> PolicySummary:
    return PolicySummary(
        id=summary.id,
        title=summary.title,
        category=summary.category,
        status=summary.status,
        analysis_trace_id=summary.analysis_trace_id,
        analysis_provider=summary.analysis_provider,
        has_analysis=summary.has_analysis,
        revision=summary.revision,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Policy artifact not found.",
    )


@router.post(
    "",
    response_model=PolicyDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Create a policy artifact in the authenticated workspace.",
    dependencies=[Depends(require_permission("policy:manage"))],
)
def create_policy(
    request: PolicyCreateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[PolicyService, Depends(get_policy_service)],
) -> PolicyDetail:
    try:
        artifact = service.create_artifact(
            workspace_id,
            PolicyCreateCommand(
                title=request.title,
                content=request.content,
                category=request.category,
                status=request.status,
                analysis_trace_id=request.analysis_trace_id,
                analysis_provider=request.analysis_provider,
                analysis=request.analysis,
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return _to_detail(artifact)


@router.get(
    "",
    response_model=PolicyListResponse,
    summary="List policy artifacts in the authenticated workspace.",
    dependencies=[Depends(require_any_permission("dashboard:view", "monitoring:view", "policy:manage"))],
)
def list_policies(
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[PolicyService, Depends(get_policy_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PolicyListResponse:
    policies, total = service.list_artifacts(workspace_id, limit=limit, offset=offset)
    return PolicyListResponse(
        policies=[_to_summary(policy) for policy in policies],
        total=total,
    )


@router.get(
    "/{policy_id}/export",
    summary="Download a policy implementation brief.",
    dependencies=[Depends(require_any_permission("dashboard:view", "monitoring:view", "policy:manage", "optimize:run"))],
)
def export_policy(
    policy_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[PolicyService, Depends(get_policy_service)],
    format: Annotated[Literal["pdf", "docx", "markdown", "json"], Query()] = "pdf",
) -> Response:
    artifact = service.get_artifact(workspace_id, policy_id)
    if artifact is None:
        raise _not_found()
    safe_title = "".join(char if char.isalnum() or char in "-_" else "_" for char in artifact.title).strip("_") or "policy"
    safe_name = f"{safe_title}_policy_brief"
    if format == "markdown":
        return Response(policy_markdown(artifact), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="{safe_name}.md"'})
    if format == "docx":
        return Response(policy_docx(artifact), media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": f'attachment; filename="{safe_name}.docx"'})
    if format == "json":
        payload = {"policy": _to_detail(artifact).model_dump(mode="json"), "brief_markdown": policy_markdown(artifact)}
        return Response(json.dumps(payload, indent=2, ensure_ascii=False), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{safe_name}.json"'})
    return Response(policy_pdf(artifact), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'})


@router.get(
    "/{policy_id}",
    response_model=PolicyDetail,
    summary="Load one policy artifact.",
    dependencies=[Depends(require_any_permission("dashboard:view", "monitoring:view", "policy:manage", "optimize:run"))],
)
def get_policy(
    policy_id: str,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[PolicyService, Depends(get_policy_service)],
) -> PolicyDetail:
    artifact = service.get_artifact(workspace_id, policy_id)
    if artifact is None:
        raise _not_found()
    return _to_detail(artifact)


@router.patch(
    "/{policy_id}",
    response_model=PolicyDetail,
    summary="Update a policy artifact and increment its revision.",
    dependencies=[Depends(require_any_permission("policy:manage", "monitoring:view", "optimize:run"))],
)
def update_policy(
    policy_id: str,
    request: PolicyUpdateRequest,
    workspace_id: Annotated[str, Depends(get_workspace_id)],
    service: Annotated[PolicyService, Depends(get_policy_service)],
) -> PolicyDetail:
    try:
        artifact = service.update_artifact(
            workspace_id,
            policy_id,
            PolicyUpdateCommand(
                title=request.title,
                content=request.content,
                category=request.category,
                status=request.status,
                analysis=request.analysis,
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if artifact is None:
        raise _not_found()
    return _to_detail(artifact)
