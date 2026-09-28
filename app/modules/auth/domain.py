"""Auth domain types."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str
    name: str
    workspace_id: str
    role: str = "viewer"
    permissions: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoredUser:
    id: str
    email: str
    name: str
    workspace_id: str
    password_hash: str
    role: str = "viewer"
    permissions: tuple[str, ...] = ()
    is_active: bool = True
