from __future__ import annotations

from typing import Annotated

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from ...dependencies import get_auth_service, get_current_user, has_permission
from .domain import AuthUser, StoredUser
from .schemas import (
    LoginRequest,
    LoginResponse,
    ManagedUserResponse,
    UserCreateRequest,
    UserResponse,
    UserUpdateRequest,
)
from .service import AuthService
from .users import hash_password, permissions_for_role

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_user_response(user: AuthUser) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        workspace_id=user.workspace_id,
        role=user.role,
        permissions=list(user.permissions),
    )


def _to_managed_response(user: StoredUser) -> ManagedUserResponse:
    return ManagedUserResponse(
        id=user.id, email=user.email, name=user.name, workspace_id=user.workspace_id,
        role=user.role, permissions=list(user.permissions), is_active=user.is_active,
    )


def _require_user_management(user: AuthUser) -> None:
    if not has_permission(user, "users:manage"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User management permission required.")


@router.post("/login", response_model=LoginResponse)
def login(
    request: LoginRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> LoginResponse:
    result = service.login(request.email, request.password)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    token, user = result
    return LoginResponse(access_token=token, user=_to_user_response(user))


@router.get("/me", response_model=UserResponse)
def me(user: Annotated[AuthUser, Depends(get_current_user)]) -> UserResponse:
    return _to_user_response(user)


@router.get("/users", response_model=list[ManagedUserResponse])
def list_users(
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> list[ManagedUserResponse]:
    _require_user_management(user)
    return [_to_managed_response(item) for item in service.list_users(user.workspace_id)]


@router.post("/users", response_model=ManagedUserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    request: UserCreateRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> ManagedUserResponse:
    _require_user_management(user)
    permissions = tuple(request.permissions if request.permissions is not None else permissions_for_role(request.role))
    try:
        created = service.create_user(StoredUser(
            id=uuid4().hex,
            email=request.email,
            name=request.name.strip(),
            workspace_id=user.workspace_id,
            password_hash=hash_password(request.password),
            role=request.role,
            permissions=permissions,
        ))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _to_managed_response(created)


@router.patch("/users/{user_id}", response_model=ManagedUserResponse)
def update_user(
    user_id: str,
    request: UserUpdateRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> ManagedUserResponse:
    _require_user_management(user)
    changes = request.model_dump(exclude_unset=True)
    if "role" in changes and "permissions" not in changes:
        changes["permissions"] = list(permissions_for_role(changes["role"]))
    if user_id == user.id and (
        changes.get("is_active") is False
        or changes.get("role", user.role) != "admin"
        or "users:manage" not in changes.get("permissions", user.permissions)
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot remove your own administrator access.")
    updated = service.update_user(user.workspace_id, user_id, changes)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found in this workspace.")
    return _to_managed_response(updated)
