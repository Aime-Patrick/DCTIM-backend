from __future__ import annotations

from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .domain import AuthUser, StoredUser
from .models import WorkspaceUser
from .users import get_user_store


class UserRepository:
    """Database-backed users; existing local demo accounts seed an empty table."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        with self._session_factory() as session:
            if session.scalar(select(func.count(WorkspaceUser.id))) or 0:
                return
            for user in get_user_store().all_users():
                session.add(WorkspaceUser(
                    id=user.id,
                    email=user.email,
                    name=user.name,
                    workspace_id=user.workspace_id,
                    password_hash=user.password_hash,
                    role=user.role,
                    permissions=list(user.permissions),
                    is_active=True,
                ))
            session.commit()

    def authenticate(self, email: str, password: str) -> AuthUser | None:
        with self._session_factory() as session:
            row = session.scalar(select(WorkspaceUser).where(func.lower(WorkspaceUser.email) == email.strip().lower()))
            if row is None or not row.is_active:
                return None
            from .users import verify_password
            if not verify_password(password, row.password_hash):
                return None
            return _to_auth_user(row)

    def get_by_id(self, user_id: str) -> AuthUser | None:
        with self._session_factory() as session:
            row = session.get(WorkspaceUser, user_id)
            return _to_auth_user(row) if row is not None and row.is_active else None

    def list_workspace(self, workspace_id: str) -> list[StoredUser]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(WorkspaceUser).where(WorkspaceUser.workspace_id == workspace_id)
                .order_by(WorkspaceUser.email)
            ).all()
            return [_to_stored_user(row) for row in rows]

    def create_user(self, user: StoredUser) -> StoredUser:
        with self._session_factory() as session:
            row = WorkspaceUser(
                id=user.id or uuid4().hex,
                email=user.email.lower(),
                name=user.name,
                workspace_id=user.workspace_id,
                password_hash=user.password_hash,
                role=user.role,
                permissions=list(user.permissions),
                is_active=user.is_active,
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ValueError("A user with that email already exists.") from exc
            return _to_stored_user(row)

    def update_user(self, workspace_id: str, user_id: str, changes: dict) -> StoredUser | None:
        with self._session_factory() as session:
            row = session.scalar(select(WorkspaceUser).where(
                WorkspaceUser.id == user_id,
                WorkspaceUser.workspace_id == workspace_id,
            ))
            if row is None:
                return None
            for key, value in changes.items():
                setattr(row, key, list(value) if key == "permissions" else value)
            session.commit()
            return _to_stored_user(row)


def _to_auth_user(row: WorkspaceUser) -> AuthUser:
    return AuthUser(row.id, row.email, row.name, row.workspace_id, row.role, tuple(row.permissions or ()))


def _to_stored_user(row: WorkspaceUser) -> StoredUser:
    return StoredUser(
        row.id, row.email, row.name, row.workspace_id, row.password_hash,
        row.role, tuple(row.permissions or ()), row.is_active,
    )
