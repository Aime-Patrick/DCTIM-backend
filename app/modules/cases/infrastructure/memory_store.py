from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from ..application import CaseCreateCommand
from ..domain import CaseSource, TransformationCase, TransformationCaseSummary, utc_now


class InMemoryTransformationCaseStore:
    def __init__(self) -> None:
        self._cases: dict[str, TransformationCase] = {}

    def create_case(self, workspace_id: str, command: CaseCreateCommand) -> TransformationCase:
        now = utc_now()
        case = TransformationCase(
            id=uuid4().hex,
            workspace_id=workspace_id,
            title=command.title,
            problem_statement=command.problem_statement,
            desired_outcome=command.desired_outcome,
            territory=command.territory,
            population=command.population,
            time_horizon=command.time_horizon,
            decision_authority=command.decision_authority,
            status=command.status,
            success_criteria=command.success_criteria,
            indicators=command.indicators,
            constraints=command.constraints,
            metadata=dict(command.metadata or {}),
            revision=1,
            review_date=command.review_date,
            created_at=now,
            updated_at=now,
        )
        self._cases[case.id] = case
        return case

    def list_cases(
        self, workspace_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[list[TransformationCaseSummary], int]:
        owned = [case for case in self._cases.values() if case.workspace_id == workspace_id]
        owned.sort(key=lambda case: (case.updated_at, case.id), reverse=True)
        summaries = [_to_summary(case) for case in owned[offset : offset + limit]]
        return summaries, len(owned)

    def get_case(self, workspace_id: str, case_id: str) -> TransformationCase | None:
        case = self._cases.get(case_id)
        if case is None or case.workspace_id != workspace_id:
            return None
        return case

    def update_case(
        self, workspace_id: str, case_id: str, changes: dict[str, Any]
    ) -> TransformationCase | None:
        case = self.get_case(workspace_id, case_id)
        if case is None:
            return None
        updated = replace(
            case,
            **changes,
            revision=case.revision + 1,
            updated_at=utc_now(),
        )
        self._cases[case_id] = updated
        return updated

    def link_source(
        self, workspace_id: str, case_id: str, document_id: str, source_role: str
    ) -> CaseSource | None:
        case = self.get_case(workspace_id, case_id)
        if case is None:
            return None
        if any(source.document_id == document_id for source in case.sources):
            raise ValueError("document is already linked to this case")
        source = CaseSource(
            id=uuid4().hex,
            case_id=case_id,
            document_id=document_id,
            source_role=source_role,
            linked_at=utc_now(),
        )
        self._cases[case_id] = replace(
            case,
            sources=case.sources + (source,),
            updated_at=utc_now(),
        )
        return source


def _to_summary(case: TransformationCase) -> TransformationCaseSummary:
    return TransformationCaseSummary(
        id=case.id,
        title=case.title,
        territory=case.territory,
        status=case.status,
        revision=case.revision,
        source_count=len(case.sources),
        created_at=case.created_at,
        updated_at=case.updated_at,
    )
