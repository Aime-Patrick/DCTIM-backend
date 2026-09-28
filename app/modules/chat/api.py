"""Chat history API: per-user conversations and messages.

Every endpoint is scoped to the authenticated user's identity
(``user.id``) extracted from the verified JWT — never from a request body
or header.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...dependencies import get_chat_service, get_current_user, require_permission
from ..auth.domain import AuthUser
from .application import ChatService
from .domain import ConversationSummary as DomainConversationSummary
from .schemas import (
    AppendMessagesRequest,
    AppendMessagesResponse,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationListResponse,
    ConversationRenameRequest,
    ConversationSummary,
    MessageOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])


def _message_out(message) -> MessageOut:
    return MessageOut(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        prompt=message.prompt,
        trace_id=message.trace_id,
        metadata=message.metadata,
    )


def _summary_out(summary: DomainConversationSummary) -> ConversationSummary:
    return ConversationSummary(
        id=summary.id,
        title=summary.title,
        message_count=summary.message_count,
        last_message_preview=summary.last_message_preview,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List the authenticated user's conversations, newest first.",
    dependencies=[Depends(require_permission("prompt:use"))],
)
def list_conversations(
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ConversationListResponse:
    conversations = service.list_conversations(user.id, limit)
    return ConversationListResponse(
        conversations=[_summary_out(c) for c in conversations],
        total=len(conversations),
    )


@router.post(
    "/conversations",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation for the authenticated user.",
    dependencies=[Depends(require_permission("prompt:use"))],
)
def create_conversation(
    request: ConversationCreateRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ConversationSummary:
    conversation = service.create_conversation(user.id, user.workspace_id, request.title)
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        message_count=0,
        last_message_preview=None,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Load one conversation with all of its messages.",
    dependencies=[Depends(require_permission("prompt:use"))],
)
def get_conversation(
    conversation_id: str,
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ConversationDetail:
    detail = service.get_conversation(user.id, conversation_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )
    return ConversationDetail(
        id=detail.id,
        title=detail.title,
        created_at=detail.created_at,
        updated_at=detail.updated_at,
        messages=[_message_out(m) for m in detail.messages],
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationSummary,
    summary="Rename a conversation.",
    dependencies=[Depends(require_permission("prompt:use"))],
)
def rename_conversation(
    conversation_id: str,
    request: ConversationRenameRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ConversationSummary:
    if not service.rename_conversation(user.id, conversation_id, request.title):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )
    detail = service.get_conversation(user.id, conversation_id)
    assert detail is not None
    return ConversationSummary(
        id=detail.id,
        title=detail.title,
        message_count=len(detail.messages),
        last_message_preview=detail.messages[-1].content if detail.messages else None,
        created_at=detail.created_at,
        updated_at=detail.updated_at,
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation and its messages.",
    dependencies=[Depends(require_permission("prompt:use"))],
)
def delete_conversation(
    conversation_id: str,
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> None:
    if not service.delete_conversation(user.id, conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=AppendMessagesResponse,
    summary="Append one user turn (user + assistant messages) to a conversation.",
    dependencies=[Depends(require_permission("prompt:use"))],
)
def append_messages(
    conversation_id: str,
    request: AppendMessagesRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> AppendMessagesResponse:
    messages = service.append_messages(
        user.id,
        user.workspace_id,
        conversation_id,
        request.messages,
    )
    if messages is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )
    return AppendMessagesResponse(
        appended=len(messages),
        messages=[_message_out(m) for m in messages],
    )
