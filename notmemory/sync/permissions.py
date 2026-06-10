# notmemory/sync/permissions.py
"""
Permission system for multi-agent notmemory access.

Controls which agents can read/write/admin which memory banks.

Access levels (ordered):
  none  → no access
  read  → recall only
  write → retain + recall
  admin → retain + recall + rollback + forget
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class AccessLevel(StrEnum):
    """Ordered access levels for memory bank permissions."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

    def allows(self, required: AccessLevel) -> bool:
        """Return True if this level satisfies the required level."""
        order = [
            AccessLevel.NONE,
            AccessLevel.READ,
            AccessLevel.WRITE,
            AccessLevel.ADMIN,
        ]
        return order.index(self) >= order.index(required)


class AgentPermission:
    """Permission record for one agent on one bank."""

    def __init__(
        self,
        agent_id: str,
        bank_id: str,
        access_level: AccessLevel,
    ) -> None:
        self.agent_id = agent_id
        self.bank_id = bank_id
        self.access_level = AccessLevel(access_level)

    def can_read(self) -> bool:
        return self.access_level.allows(AccessLevel.READ)

    def can_write(self) -> bool:
        return self.access_level.allows(AccessLevel.WRITE)

    def can_admin(self) -> bool:
        return self.access_level.allows(AccessLevel.ADMIN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "bank_id": self.bank_id,
            "access_level": self.access_level.value,
        }

    def __repr__(self) -> str:
        return (
            f"<AgentPermission agent={self.agent_id!r}"
            f" bank={self.bank_id!r}"
            f" level={self.access_level.value!r}>"
        )


class PermissionStore:
    """
    In-memory permission store for multi-agent access control.

    Maps (agent_id, bank_id) → AccessLevel.
    Supports wildcard bank_id="*" for global permissions.

    Usage:
        store = PermissionStore()
        store.grant("agent-1", "facts", AccessLevel.WRITE)
        store.grant("agent-2", "*", AccessLevel.READ)

        store.check("agent-1", "facts", AccessLevel.WRITE)  # True
        store.check("agent-2", "facts", AccessLevel.ADMIN)  # False
    """

    def __init__(self) -> None:
        # (agent_id, bank_id) → AccessLevel
        self._permissions: dict[tuple[str, str], AccessLevel] = {}

    def grant(
        self,
        agent_id: str,
        bank_id: str,
        access_level: AccessLevel,
    ) -> None:
        """Grant an agent access to a bank at the given level."""
        self._permissions[(agent_id, bank_id)] = AccessLevel(access_level)

    def revoke(self, agent_id: str, bank_id: str) -> None:
        """Revoke all access for an agent on a bank."""
        self._permissions.pop((agent_id, bank_id), None)
        self._permissions.pop((agent_id, "*"), None)

    def check(
        self,
        agent_id: str,
        bank_id: str,
        required: AccessLevel,
    ) -> bool:
        """
        Return True if agent has at least the required access level.

        Checks specific bank permission first, then wildcard "*".
        """
        # Specific bank permission
        level = self._permissions.get((agent_id, bank_id))
        if level is not None:
            return level.allows(required)

        # Wildcard permission
        wildcard = self._permissions.get((agent_id, "*"))
        if wildcard is not None:
            return wildcard.allows(required)

        return False

    def get_permission(self, agent_id: str, bank_id: str) -> AgentPermission:
        """Return AgentPermission for agent+bank (NONE if not set)."""
        level = (
            self._permissions.get((agent_id, bank_id))
            or self._permissions.get((agent_id, "*"))
            or AccessLevel.NONE
        )

        return AgentPermission(agent_id, bank_id, level)

    def list_permissions(self, agent_id: str | None = None) -> list[AgentPermission]:
        """List all permissions, optionally filtered by agent_id."""
        results = []
        for (aid, bid), level in self._permissions.items():
            if agent_id is None or aid == agent_id:
                results.append(AgentPermission(aid, bid, level))
        return results

    def all_agents(self) -> list[str]:
        """Return all agent IDs that have any permission."""
        return list({aid for aid, _ in self._permissions})


__all__ = ["AccessLevel", "AgentPermission", "PermissionStore"]
