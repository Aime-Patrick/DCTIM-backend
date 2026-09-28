"""Chat domain types.

Conversations belong to a single user (``user_id`` == the verified JWT
subject) and are additionally tagged with the user's ``workspace_id`` so
cross-module queries never have to guess who owns what.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# Server-side default title; replaced with the first user message on append.
DEFAULT_TITLE = "New conversation"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ChatMessage:
    id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime = field(default_factory=utc_now)
    prompt: str | None = None
    trace_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Conversation:
    id: str
    user_id: str
    workspace_id: str
    title: str
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    message_count: int
    last_message_preview: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationDetail:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage] = field(default_factory=list)