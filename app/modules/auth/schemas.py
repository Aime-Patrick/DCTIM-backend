from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .users import PERMISSION_KEYS

UserRole = Literal["admin", "policy_manager", "analyst", "viewer"]


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    workspace_id: str
    role: UserRole = "viewer"
    permissions: list[str] = Field(default_factory=list)


class ManagedUserResponse(UserResponse):
    is_active: bool


class UserCreateRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=8, max_length=200)
    role: UserRole = "analyst"
    permissions: list[str] | None = Field(default=None, max_length=len(PERMISSION_KEYS))

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name is required.")
        return value

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or "." not in value.rsplit("@", 1)[-1]:
            raise ValueError("Enter a valid email address.")
        return value

    @field_validator("permissions")
    @classmethod
    def valid_permissions(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(item not in PERMISSION_KEYS for item in value):
            raise ValueError("Unknown permission.")
        return list(dict.fromkeys(value)) if value is not None else None


class UserUpdateRequest(BaseModel):
    role: UserRole | None = None
    permissions: list[str] | None = Field(default=None, max_length=len(PERMISSION_KEYS))
    is_active: bool | None = None

    @field_validator("permissions")
    @classmethod
    def valid_permissions(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(item not in PERMISSION_KEYS for item in value):
            raise ValueError("Unknown permission.")
        return list(dict.fromkeys(value)) if value is not None else None

    @model_validator(mode="after")
    def require_update(self) -> "UserUpdateRequest":
        if not self.model_fields_set or any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("Provide at least one non-null field to update.")
        return self


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
