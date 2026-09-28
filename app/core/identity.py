"""Request identity helpers.

Workspace isolation is enforced by deriving ``workspace_id`` from a verified
JWT (see ``dependencies.get_workspace_id``). Clients cannot pick another
workspace via headers or body fields.
"""
from __future__ import annotations

# Re-export so existing imports of ``core.identity.get_workspace_id`` keep working.
from ..dependencies import get_current_user, get_workspace_id

__all__ = ["get_current_user", "get_workspace_id"]
