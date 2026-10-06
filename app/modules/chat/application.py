"""ChatService — application logic for the chat history module.

The service wraps either the in-memory or the SQLAlchemy store and owns the
few business decisions (server-assigned ids, auto-titling of new
conversations) so the API layer stays thin.
"""
from __future__ import annotations

from uuid import uuid4

from .domain import (
    ChatMessage,
    Conversation,
    ConversationDetail,
    ConversationSummary,
    DEFAULT_TITLE,
)
from .schemas import MessageIn

_TITLE_SLICE = 60


class ChatService:
    def __init__(self, store) -> None:
        self._store = store

    def create_conversation(
        self,
        user_id: str,
        workspace_id: str,
        title: str | None = None,
    ) -> Conversation:
        return self._store.create_conversation(user_id, workspace_id, title)

    def list_conversations(self, user_id: str, limit: int = 50) -> list[ConversationSummary]:
        return self._store.list_conversations(user_id, limit)

    def get_conversation(self, user_id: str, conversation_id: str) -> ConversationDetail | None:
        detail = self._store.get_conversation(user_id, conversation_id)
        if detail is None:
            return None
        return ConversationDetail(
            id=detail.id,
            title=detail.title,
            created_at=detail.created_at,
            updated_at=detail.updated_at,
            messages=[
                message
                for message in detail.messages
                if not bool((message.metadata or {}).get("is_error"))
            ],
        )

    def append_messages(
        self,
        user_id: str,
        workspace_id: str,
        conversation_id: str,
        messages: list[MessageIn],
    ) -> list[ChatMessage] | None:
        # Provider failures and other transient UI notices are not conversation
        # content. Keep this server-side guard even if an older client submits
        # an error message directly.
        messages = [
            message for message in messages
            if not bool((message.metadata or {}).get("is_error"))
        ]
        if not messages:
            return []

        created = [
            ChatMessage(
                id=uuid4().hex,
                conversation_id=conversation_id,
                user_id=user_id,
                role=message.role.value,
                content=message.content.strip(),
                prompt=message.prompt,
                trace_id=message.trace_id,
                metadata=message.metadata,
            )
            for message in messages
        ]
        stored = self._store.append_messages(user_id, conversation_id, created)
        if stored is None:
            return None

        self._maybe_auto_title(user_id, workspace_id, conversation_id)
        return created

    def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> bool:
        if not title.strip():
            return False
        return self._store.update_title(user_id, conversation_id, title)

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        return self._store.delete_conversation(user_id, conversation_id)

    def _maybe_auto_title(
        self,
        user_id: str,
        workspace_id: str,
        conversation_id: str,
    ) -> None:
        detail = self._store.get_conversation(user_id, conversation_id)
        if detail is None or detail.title != DEFAULT_TITLE:
            return
        first_user = next((m.content for m in detail.messages if m.role == "user"), None)
        if not first_user:
            return
        title = " ".join(first_user.split())[:_TITLE_SLICE]
        if title:
            self._store.update_title(user_id, conversation_id, title)
