"""In-memory chat store used when no database is configured.

Mimics the interface of the SQLAlchemy-backed ``ChatRepository`` so the
service and API layer do not care which store is wired in.
"""
from __future__ import annotations

from uuid import uuid4

from ..domain import (
    ChatMessage,
    Conversation,
    ConversationDetail,
    ConversationSummary,
    DEFAULT_TITLE,
    utc_now,
)


class InMemoryChatStore:
    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._messages: dict[str, list[ChatMessage]] = {}

    def create_conversation(
        self,
        user_id: str,
        workspace_id: str,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            id=uuid4().hex,
            user_id=user_id,
            workspace_id=workspace_id,
            title=(title or DEFAULT_TITLE).strip()[:200],
        )
        self._conversations[conversation.id] = conversation
        self._messages[conversation.id] = []
        return conversation

    def list_conversations(self, user_id: str, limit: int = 50) -> list[ConversationSummary]:
        owned = [
            c for c in self._conversations.values() if c.user_id == user_id
        ]
        owned.sort(key=lambda c: c.updated_at, reverse=True)
        summaries = []
        for conv in owned[:limit]:
            msgs = self._messages.get(conv.id, [])
            summaries.append(
                ConversationSummary(
                    id=conv.id,
                    title=conv.title,
                    message_count=len(msgs),
                    last_message_preview=_preview(msgs[-1].content) if msgs else None,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                )
            )
        return summaries

    def get_conversation(self, user_id: str, conversation_id: str) -> ConversationDetail | None:
        conv = self._conversations.get(conversation_id)
        if conv is None or conv.user_id != user_id:
            return None
        msgs = self._messages.get(conversation_id, [])
        return ConversationDetail(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=[*msgs],
        )

    def append_messages(
        self,
        user_id: str,
        conversation_id: str,
        messages: list[ChatMessage],
    ) -> list[ChatMessage] | None:
        conv = self._conversations.get(conversation_id)
        if conv is None or conv.user_id != user_id:
            return None
        conv_msgs = self._messages[conversation_id]
        for msg in messages:
            conv_msgs.append(msg)
        self._conversations[conversation_id] = Conversation(
            id=conv.id,
            user_id=conv.user_id,
            workspace_id=conv.workspace_id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=utc_now(),
        )
        return list(messages)

    def update_title(self, user_id: str, conversation_id: str, title: str) -> bool:
        conv = self._conversations.get(conversation_id)
        if conv is None or conv.user_id != user_id:
            return False
        self._conversations[conversation_id] = Conversation(
            id=conv.id,
            user_id=conv.user_id,
            workspace_id=conv.workspace_id,
            title=title.strip()[:200],
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        )
        return True

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        conv = self._conversations.get(conversation_id)
        if conv is None or conv.user_id != user_id:
            return False
        del self._conversations[conversation_id]
        self._messages.pop(conversation_id, None)
        return True


def _preview(content: str, limit: int = 120) -> str:
    content = " ".join(content.split())
    return content[:limit] + ("…" if len(content) > limit else "")