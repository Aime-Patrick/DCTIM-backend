"""Auth application service."""
from __future__ import annotations

from .domain import AuthUser, StoredUser
from .tokens import create_access_token
from .users import UserStore


class AuthService:
    def __init__(
        self,
        users: UserStore,
        *,
        jwt_secret: str,
        jwt_expires_minutes: int,
    ) -> None:
        self._users = users
        self._jwt_secret = jwt_secret
        self._jwt_expires_minutes = jwt_expires_minutes

    def login(self, email: str, password: str) -> tuple[str, AuthUser] | None:
        user = self._users.authenticate(email, password)
        if user is None:
            return None
        token = create_access_token(
            user,
            secret=self._jwt_secret,
            expires_minutes=self._jwt_expires_minutes,
        )
        return token, user

    def user_from_token(self, token: str) -> AuthUser:
        from .tokens import decode_access_token

        claims = decode_access_token(token, secret=self._jwt_secret)
        # Prefer live store record so revoked users disappear after logout/restart.
        live = self._users.get_by_id(claims.id)
        if live is None:
            raise ValueError("user no longer exists")
        return live

    def list_users(self, workspace_id: str) -> list[StoredUser]:
        return self._users.list_workspace(workspace_id)

    def create_user(self, user: StoredUser) -> StoredUser:
        return self._users.create_user(user)

    def update_user(self, workspace_id: str, user_id: str, changes: dict) -> StoredUser | None:
        return self._users.update_user(workspace_id, user_id, changes)
