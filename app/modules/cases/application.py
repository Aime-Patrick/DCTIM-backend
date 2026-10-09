from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .domain import CaseSource, TransformationCase, TransformationCaseSummary


@dataclass(frozen=True)
class CaseCreateCommand:
    title: str
    problem_statement: str
    desired_outcome: str
    territory: str | None = None
    population: str | None = None
    time_horizon: str | None = None
    decision_authority: str | None = None
    status: str = "draft"
    success_criteria: tuple[str, ...] = ()
    indicators: tuple[dict[str, Any], ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    review_date: datetime | None = None


@dataclass(frozen=True)
class CaseUpdateCommand:
    title: str | None = None
    problem_statement: str | None = None
    desired_outcome: str | None = None
    territory: str | None = None
    population: str | None = None
    time_horizon: str | None = None
    decision_authority: str | None = None
    status: str | None = None
    success_criteria: tuple[str, ...] | None = None
    indicators: tuple[dict[str, Any], ...] | None = None
    constraints: tuple[str, ...] | None = None
    metadata: dict[str, Any] | None = None
    review_date: datetime | None = None


class CaseStore(Protocol):
    def create_case(self, workspace_id: str, command: CaseCreateCommand) -> TransformationCase: ...

    def list_cases(
        self, workspace_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[list[TransformationCaseSummary], int]: ...

    def get_case(self, workspace_id: str, case_id: str) -> TransformationCase | None: ...

    def update_case(
        self, workspace_id: str, case_id: str, changes: dict[str, Any]
    ) -> TransformationCase | None: ...

    def link_source(
        self, workspace_id: str, case_id: str, document_id: str, source_role: str
    ) -> CaseSource | None: ...


class TransformationCaseService:
    def __init__(self, store: CaseStore) -> None:
        self._store = store

    def create_case(self, workspace_id: str, command: CaseCreateCommand) -> TransformationCase:
        return self._store.create_case(
            workspace_id,
            CaseCreateCommand(
                title=_required_text(command.title, "title"),
                problem_statement=_required_text(command.problem_statement, "problem_statement"),
                desired_outcome=_required_text(command.desired_outcome, "desired_outcome"),
                territory=_optional_text(command.territory),
                population=_optional_text(command.population),
                time_horizon=_optional_text(command.time_horizon),
                decision_authority=_optional_text(command.decision_authority),
                status=command.status,
                success_criteria=_clean_texts(command.success_criteria),
                indicators=tuple(dict(item) for item in command.indicators),
                constraints=_clean_texts(command.constraints),
                metadata=dict(command.metadata or {}),
                review_date=command.review_date,
            ),
        )

    def list_cases(
        self, workspace_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[list[TransformationCaseSummary], int]:
        return self._store.list_cases(workspace_id, limit=limit, offset=offset)

    def get_case(self, workspace_id: str, case_id: str) -> TransformationCase | None:
        return self._store.get_case(workspace_id, case_id)

    def update_case(
        self, workspace_id: str, case_id: str, command: CaseUpdateCommand
    ) -> TransformationCase | None:
        changes: dict[str, Any] = {}
        for field_name in (
            "title",
            "problem_statement",
            "desired_outcome",
            "territory",
            "population",
            "time_horizon",
            "decision_authority",
            "status",
            "success_criteria",
            "indicators",
            "constraints",
            "metadata",
            "review_date",
        ):
            value = getattr(command, field_name)
            if value is None:
                continue
            if field_name in {"title", "problem_statement", "desired_outcome"}:
                changes[field_name] = _required_text(value, field_name)
            elif field_name in {"territory", "population", "time_horizon", "decision_authority"}:
                changes[field_name] = _optional_text(value)
            elif field_name in {"success_criteria", "constraints"}:
                changes[field_name] = _clean_texts(value)
            elif field_name == "indicators":
                changes[field_name] = tuple(dict(item) for item in value)
            elif field_name == "metadata":
                changes[field_name] = dict(value)
            else:
                changes[field_name] = value
        if not changes:
            return self.get_case(workspace_id, case_id)
        return self._store.update_case(workspace_id, case_id, changes)

    def link_source(
        self, workspace_id: str, case_id: str, document_id: str, source_role: str
    ) -> CaseSource | None:
        normalized_document_id = _required_text(document_id, "document_id")
        normalized_role = _required_text(source_role, "source_role")
        return self._store.link_source(
            workspace_id, case_id, normalized_document_id, normalized_role
        )


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


def _clean_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value.strip())
