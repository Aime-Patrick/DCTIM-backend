"""JWT helpers for access tokens."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from .domain import AuthUser

ALGORITHM = "HS256"


def create_access_token(
    user: AuthUser,
    *,
    secret: str,
    expires_minutes: int,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user.id,
        "email": user.email,
        "name": user.name,
        "workspace_id": user.workspace_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, *, secret: str) -> AuthUser:
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise ValueError("invalid or expired token") from exc

    try:
        return AuthUser(
            id=str(payload["sub"]),
            email=str(payload["email"]),
            name=str(payload["name"]),
            workspace_id=str(payload["workspace_id"]),
        )
    except KeyError as exc:
        raise ValueError("token missing required claims") from exc
