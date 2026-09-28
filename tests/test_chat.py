"""Chat history API tests: per-user CRUD, isolation, auto-titling."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import app.dependencies as deps
from app.main import create_app
from app.modules.auth.users import get_user_store


def _client() -> TestClient:
    get_user_store.cache_clear()
    deps.get_settings.cache_clear()
    deps.get_auth_service.cache_clear()
    deps._memory_vector_store = None
    deps._memory_chat_store = None
    deps._engine = None
    deps._session_factory = None
    os.environ.pop("DATABASE_URL", None)
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hash"
    os.environ["RAG_ANSWER_PROVIDER"] = "demo"
    return TestClient(create_app())


def _login(client: TestClient, email: str, password: str) -> dict:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    return response.json()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_chat_requires_bearer_token() -> None:
    client = _client()
    assert client.get("/api/v1/chat/conversations").status_code == 401
    assert client.post(
        "/api/v1/chat/conversations",
        json={"title": "Test"},
        headers={"X-Workspace-Id": "admin"},
    ).status_code == 401


def test_create_and_append_messages() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")["access_token"]

    created = client.post(
        "/api/v1/chat/conversations",
        json={},
        headers=_auth(token),
    )
    assert created.status_code == 201
    conv = created.json()
    assert conv["title"] == "New conversation"
    assert conv["message_count"] == 0

    appended = client.post(
        f"/api/v1/chat/conversations/{conv['id']}/messages",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "What drives GDP growth in developing nations?",
                    "metadata": {"original_input": "GDP growth?"},
                },
                {
                    "role": "assistant",
                    "content": "Several structural factors…",
                    "trace_id": "trace-abc",
                },
            ]
        },
        headers=_auth(token),
    )
    assert appended.status_code == 200
    body = appended.json()
    assert body["appended"] == 2
    assert body["messages"][0]["role"] == "user"

    detail = client.get(
        f"/api/v1/chat/conversations/{conv['id']}",
        headers=_auth(token),
    ).json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["metadata"]["original_input"] == "GDP growth?"
    assert detail["messages"][1]["trace_id"] == "trace-abc"


def test_conversation_auto_titles_from_first_user_message() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")["access_token"]

    conv = client.post(
        "/api/v1/chat/conversations",
        json={},
        headers=_auth(token),
    ).json()
    client.post(
        f"/api/v1/chat/conversations/{conv['id']}/messages",
        json={
            "messages": [
                {"role": "user", "content": "Healthcare infrastructure in low-income countries"},
                {"role": "assistant", "content": "Analysis…"},
            ]
        },
        headers=_auth(token),
    )

    listed = client.get(
        "/api/v1/chat/conversations",
        headers=_auth(token),
    ).json()
    assert listed["total"] == 1
    assert listed["conversations"][0]["title"] == "Healthcare infrastructure in low-income countries"
    assert listed["conversations"][0]["message_count"] == 2
    assert listed["conversations"][0]["last_message_preview"]


def test_conversations_are_isolated_per_user() -> None:
    client = _client()
    admin = _login(client, "admin@dc-tim.ai", "admin123")["access_token"]
    analyst = _login(client, "analyst@dc-tim.ai", "analyst123")["access_token"]

    conv = client.post(
        "/api/v1/chat/conversations",
        json={"title": "Admin secret"},
        headers=_auth(admin),
    ).json()
    client.post(
        f"/api/v1/chat/conversations/{conv['id']}/messages",
        json={"messages": [{"role": "user", "content": "hidden"}, {"role": "assistant", "content": "hidden"}]},
        headers=_auth(admin),
    )

    # Analyst's list does not include the admin conversation.
    listed = client.get(
        "/api/v1/chat/conversations",
        headers=_auth(analyst),
    ).json()
    assert listed["total"] == 0

    # Analyst cannot read, append to, rename, or delete it.
    assert client.get(
        f"/api/v1/chat/conversations/{conv['id']}",
        headers=_auth(analyst),
    ).status_code == 404
    assert client.post(
        f"/api/v1/chat/conversations/{conv['id']}/messages",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=_auth(analyst),
    ).status_code == 404
    assert client.patch(
        f"/api/v1/chat/conversations/{conv['id']}",
        json={"title": "hacked"},
        headers=_auth(analyst),
    ).status_code == 404
    assert client.delete(
        f"/api/v1/chat/conversations/{conv['id']}",
        headers=_auth(analyst),
    ).status_code == 404

    # Admin still owns it.
    assert client.get(
        f"/api/v1/chat/conversations/{conv['id']}",
        headers=_auth(admin),
    ).status_code == 200


def test_rename_and_delete_conversation() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")["access_token"]

    conv = client.post(
        "/api/v1/chat/conversations",
        json={},
        headers=_auth(token),
    ).json()

    renamed = client.patch(
        f"/api/v1/chat/conversations/{conv['id']}",
        json={"title": "Budget review"},
        headers=_auth(token),
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Budget review"

    deleted = client.delete(
        f"/api/v1/chat/conversations/{conv['id']}",
        headers=_auth(token),
    )
    assert deleted.status_code == 204

    assert client.get(
        f"/api/v1/chat/conversations/{conv['id']}",
        headers=_auth(token),
    ).status_code == 404


def test_messages_are_rejected_for_unknown_conversation() -> None:
    client = _client()
    token = _login(client, "admin@dc-tim.ai", "admin123")["access_token"]
    response = client.post(
        "/api/v1/chat/conversations/does-not-exist/messages",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=_auth(token),
    )
    assert response.status_code == 404