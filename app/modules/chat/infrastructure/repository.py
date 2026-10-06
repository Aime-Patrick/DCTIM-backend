"""ChatRepository — persists conversations and messages via SQLAlchemy.

User isolation is enforced at the query level (every SELECT/UPDATE/DELETE
filters on ``user_id``), matching the workspace-scoped RAG repository.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..domain import (
    ChatMessage,
    Conversation,
    ConversationDetail,
    ConversationSummary,
    DEFAULT_TITLE,
)
from .models import ChatConversation, ChatMessage as ChatMessageRow


class ChatRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_conversation(
        self,
        user_id: str,
        workspace_id: str,
        title: str | None = None,
    ) -> Conversation:
        row = ChatConversation(
            id=uuid4().hex,
            user_id=user_id,
            workspace_id=workspace_id,
            title=(title or DEFAULT_TITLE).strip()[:200],
        )
        self._session.add(row)
        self._session.flush()
        return Conversation(
            id=row.id,
            user_id=row.user_id,
            workspace_id=row.workspace_id,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def list_conversations(self, user_id: str, limit: int = 50) -> list[ConversationSummary]:
        rows = self._session.scalars(
            select(ChatConversation)
            .where(ChatConversation.user_id == user_id)
            .order_by(ChatConversation.updated_at.desc())
            .limit(limit)
        ).all()

        conv_ids = [row.id for row in rows]
        counts: dict[str, int] = {}
        last_content: dict[str, str] = {}
        if conv_ids:
            # Older clients may have persisted transient error bubbles. Exclude
            # them from summaries as well as conversation detail responses.
            message_rows = self._session.scalars(
                select(ChatMessageRow)
                .where(ChatMessageRow.conversation_id.in_(conv_ids))
                .order_by(ChatMessageRow.conversation_id, ChatMessageRow.ordinal.asc())
            ).all()
            valid_rows = [
                message
                for message in message_rows
                if not bool((message.metadata_ or {}).get("is_error"))
            ]
            for message in valid_rows:
                counts[message.conversation_id] = counts.get(message.conversation_id, 0) + 1
                last_content[message.conversation_id] = message.content

        return [
            ConversationSummary(
                id=row.id,
                title=row.title,
                message_count=counts.get(row.id, 0),
                last_message_preview=last_content.get(row.id) if counts.get(row.id, 0) else None,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    def get_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> ConversationDetail | None:
        row = self._session.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        )
        if row is None:
            return None

        messages = self._session.scalars(
            select(ChatMessageRow)
            .where(ChatMessageRow.conversation_id == conversation_id)
            .order_by(ChatMessageRow.ordinal.asc())
        ).all()

        return ConversationDetail(
            id=row.id,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
            messages=[
                ChatMessage(
                    id=m.id,
                    conversation_id=m.conversation_id,
                    user_id=m.user_id,
                    role=m.role,
                    content=m.content,
                    prompt=m.prompt,
                    trace_id=m.trace_id,
                    metadata=dict(m.metadata_ or {}),
                    created_at=m.created_at,
                )
                for m in messages
            ],
        )

    def append_messages(
        self,
        user_id: str,
        conversation_id: str,
        messages: list[ChatMessage],
    ) -> list[ChatMessage] | None:
        exists = self._session.scalar(
            select(ChatConversation.id).where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        )
        if not exists:
            return None

        next_ordinal = (
            self._session.scalar(
                select(func.max(ChatMessageRow.ordinal)).where(
                    ChatMessageRow.conversation_id == conversation_id
                )
            )
            or 0
        )

        rows = [
            ChatMessageRow(
                id=msg.id,
                conversation_id=conversation_id,
                user_id=user_id,
                role=msg.role,
                content=msg.content,
                prompt=msg.prompt,
                trace_id=msg.trace_id,
                metadata_=msg.metadata,
                ordinal=next_ordinal + idx,
            )
            for idx, msg in enumerate(messages)
        ]
        self._session.add_all(rows)
        self._session.execute(
            update(ChatConversation)
            .where(ChatConversation.id == conversation_id)
            .values(updated_at=func.now())
        )
        self._session.flush()
        return list(messages)

    def update_title(self, user_id: str, conversation_id: str, title: str) -> bool:
        result = self._session.execute(
            update(ChatConversation)
            .where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
            .values(title=title.strip()[:200])
        )
        return result.rowcount > 0

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        result = self._session.execute(
            ChatConversation.__table__.delete().where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        )
        return result.rowcount > 0
