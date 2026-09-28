"""Seeded user store for local development.

Passwords are bcrypt-hashed at process start. Replace with a database-backed
user repository when the product outgrows demo accounts.
"""
from __future__ import annotations

import re
from functools import lru_cache

import bcrypt

from .domain import AuthUser, StoredUser

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def workspace_from_email(email: str) -> str:
    local = email.split("@", 1)[0].lower()
    slug = re.sub(r"[^a-z0-9-]", "-", local)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "workspace"


def hash_password(password: str) -> str:
    # rounds=10 is enough for local demo accounts; production should use a DB-backed store.
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# Demo accounts (same credentials previously used by the frontend mock).
_SEED: tuple[tuple[str, str, str, str], ...] = (
    ("admin", "admin@dc-tim.ai", "Admin User", "admin123"),
    ("analyst", "analyst@dc-tim.ai", "Policy Analyst", "analyst123"),
)

PERMISSION_KEYS = (
    "dashboard:view", "policy:manage", "prompt:use", "optimize:run",
    "data:view", "data:manage", "monitoring:view", "users:manage",
)

ROLE_PERMISSIONS = {
    "admin": PERMISSION_KEYS,
    "policy_manager": ("dashboard:view", "policy:manage", "prompt:use", "optimize:run", "data:view", "data:manage", "monitoring:view"),
    "analyst": ("dashboard:view", "prompt:use", "optimize:run", "data:view", "monitoring:view"),
    "viewer": ("dashboard:view", "monitoring:view"),
}


def permissions_for_role(role: str) -> tuple[str, ...]:
    return tuple(ROLE_PERMISSIONS.get(role, ()))


@lru_cache
def get_user_store() -> "UserStore":
    users = [
        StoredUser(
            id=user_id,
            email=email.lower(),
            name=name,
            workspace_id=workspace_from_email(email),
            password_hash=hash_password(password),
            role="admin" if user_id == "admin" else "analyst",
            permissions=permissions_for_role("admin" if user_id == "admin" else "analyst"),
        )
        for user_id, email, name, password in _SEED
    ]
    return UserStore(users)


class UserStore:
    def __init__(self, users: list[StoredUser]) -> None:
        self._by_email = {user.email.lower(): user for user in users}

    def authenticate(self, email: str, password: str) -> AuthUser | None:
        normalized = email.strip().lower()
        if not _EMAIL_RE.match(normalized):
            return None
        stored = self._by_email.get(normalized)
        if stored is None or not stored.is_active or not verify_password(password, stored.password_hash):
            return None
        return AuthUser(
            id=stored.id,
            email=stored.email,
            name=stored.name,
            workspace_id=stored.workspace_id,
            role=stored.role,
            permissions=stored.permissions,
        )

    def get_by_id(self, user_id: str) -> AuthUser | None:
        for stored in self._by_email.values():
            if stored.id == user_id and stored.is_active:
                return AuthUser(
                    id=stored.id,
                    email=stored.email,
                    name=stored.name,
                    workspace_id=stored.workspace_id,
                    role=stored.role,
                    permissions=stored.permissions,
                )
        return None

    def list_workspace(self, workspace_id: str) -> list[StoredUser]:
        return sorted(
            (user for user in self._by_email.values() if user.workspace_id == workspace_id),
            key=lambda user: user.email,
        )

    def all_users(self) -> list[StoredUser]:
        return list(self._by_email.values())

    def create_user(self, user: StoredUser) -> StoredUser:
        if user.email.lower() in self._by_email:
            raise ValueError("A user with that email already exists.")
        self._by_email[user.email.lower()] = user
        return user

    def update_user(self, workspace_id: str, user_id: str, changes: dict) -> StoredUser | None:
        for email, user in self._by_email.items():
            if user.id == user_id and user.workspace_id == workspace_id:
                updated = StoredUser(**{**user.__dict__, **changes})
                self._by_email[email] = updated
                return updated
        return None
