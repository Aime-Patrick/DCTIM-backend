from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from ..domain import PolicyArtifact, PolicySummary, utc_now


class InMemoryPolicyStore:
    def __init__(self) -> None:
        self._artifacts: dict[str, PolicyArtifact] = {}

    def create_artifact(
        self,
        workspace_id: str,
        title: str,
        content: str,
        category: str,
        status: str,
        analysis_trace_id: str | None,
        analysis_provider: str | None,
        analysis: dict[str, Any] | None,
    ) -> PolicyArtifact:
        now = utc_now()
        artifact = PolicyArtifact(
            id=uuid4().hex,
            workspace_id=workspace_id,
            title=title,
            content=content,
            category=category,
            status=status,
            analysis_trace_id=analysis_trace_id,
            analysis_provider=analysis_provider,
            analysis=analysis,
            revision=1,
            created_at=now,
            updated_at=now,
        )
        self._artifacts[artifact.id] = artifact
        return artifact

    def list_artifacts(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PolicySummary], int]:
        owned = [
            artifact
            for artifact in self._artifacts.values()
            if artifact.workspace_id == workspace_id
        ]
        owned.sort(key=lambda artifact: (artifact.updated_at, artifact.id), reverse=True)
        summaries = [_to_summary(artifact) for artifact in owned[offset : offset + limit]]
        return summaries, len(owned)

    def get_artifact(self, workspace_id: str, policy_id: str) -> PolicyArtifact | None:
        artifact = self._artifacts.get(policy_id)
        if artifact is None or artifact.workspace_id != workspace_id:
            return None
        return artifact

    def update_artifact(
        self,
        workspace_id: str,
        policy_id: str,
        changes: dict[str, Any],
    ) -> PolicyArtifact | None:
        artifact = self.get_artifact(workspace_id, policy_id)
        if artifact is None:
            return None
        updated = replace(
            artifact,
            **{
                field_name: changes[field_name]
                for field_name in ("title", "content", "category", "status", "analysis")
                if field_name in changes
            },
            revision=artifact.revision + 1,
            updated_at=utc_now(),
        )
        self._artifacts[policy_id] = updated
        return updated


def _to_summary(artifact: PolicyArtifact) -> PolicySummary:
    return PolicySummary(
        id=artifact.id,
        title=artifact.title,
        category=artifact.category,
        status=artifact.status,
        analysis_trace_id=artifact.analysis_trace_id,
        analysis_provider=artifact.analysis_provider,
        has_analysis=artifact.analysis is not None,
        revision=artifact.revision,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
