from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import PolicyArtifact, PolicySummary


@dataclass(frozen=True)
class PolicyCreateCommand:
    title: str
    content: str
    category: str
    status: str = "created"
    analysis_trace_id: str | None = None
    analysis_provider: str | None = None
    analysis: dict[str, Any] | None = None


@dataclass(frozen=True)
class PolicyUpdateCommand:
    title: str | None = None
    content: str | None = None
    category: str | None = None
    status: str | None = None
    analysis: dict[str, Any] | None = None


class PolicyService:
    def __init__(self, store) -> None:
        self._store = store

    def create_artifact(
        self,
        workspace_id: str,
        command: PolicyCreateCommand,
    ) -> PolicyArtifact:
        title = _required_text(command.title, "title")
        content = _required_text(command.content, "content")
        category = _required_text(command.category, "category")
        return self._store.create_artifact(
            workspace_id=workspace_id,
            title=title,
            content=content,
            category=category,
            status=command.status,
            analysis_trace_id=_optional_text(command.analysis_trace_id),
            analysis_provider=_optional_text(command.analysis_provider),
            analysis=command.analysis,
        )

    def list_artifacts(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PolicySummary], int]:
        return self._store.list_artifacts(workspace_id, limit=limit, offset=offset)

    def get_artifact(self, workspace_id: str, policy_id: str) -> PolicyArtifact | None:
        return self._store.get_artifact(workspace_id, policy_id)

    def update_artifact(
        self,
        workspace_id: str,
        policy_id: str,
        command: PolicyUpdateCommand,
    ) -> PolicyArtifact | None:
        changes: dict[str, Any] = {}
        if command.title is not None:
            changes["title"] = _required_text(command.title, "title")
        if command.content is not None:
            changes["content"] = _required_text(command.content, "content")
        if command.category is not None:
            changes["category"] = _required_text(command.category, "category")
        if command.status is not None:
            changes["status"] = command.status
        if command.analysis is not None:
            changes["analysis"] = command.analysis
        if not changes:
            return self.get_artifact(workspace_id, policy_id)
        return self._store.update_artifact(workspace_id, policy_id, changes)


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
