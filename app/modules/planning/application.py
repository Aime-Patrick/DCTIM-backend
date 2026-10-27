from __future__ import annotations

import hashlib
from typing import Any


def policy_scenario_id(workspace_id: str, policy_id: str) -> str:
    digest = hashlib.sha256(f"{workspace_id}:{policy_id}".encode()).hexdigest()[:32]
    return f"ai-policy-{digest}"


def merge_policy_features(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Upsert generated proposals by stable proposal id without duplicating them.

    A proposal edited by a planner is preserved on later AI syncs. Other scenario
    features are retained unchanged.
    """
    merged = list(existing)
    positions = {
        feature.get("properties", {}).get("ai_proposal_id"): index
        for index, feature in enumerate(merged)
        if isinstance(feature, dict)
        and isinstance(feature.get("properties", {}).get("ai_proposal_id"), str)
    }
    for feature in incoming:
        proposal_id = feature["properties"]["ai_proposal_id"]
        position = positions.get(proposal_id)
        if position is None:
            positions[proposal_id] = len(merged)
            merged.append(feature)
        elif not merged[position].get("properties", {}).get("user_modified", False):
            merged[position] = feature
    return merged
