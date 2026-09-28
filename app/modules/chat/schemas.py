"""Request/response schemas for the chat history API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageIn(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1, max_length=400_000)
    prompt: str | None = Field(default=None, max_length=100_000)
    trace_id: str | None = Field(default=None, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageOut(BaseModel):
    id: str
    role: ChatRole
    content: str
    created_at: datetime
    prompt: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationSummary(BaseModel):
    id: str
    title: str
    message_count: int
    last_message_preview: str | None = None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
    total: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut] = Field(default_factory=list)


class AppendMessagesRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=100)


class AppendMessagesResponse(BaseModel):
    appended: int
    messages: list[MessageOut] = Field(default_factory=list)