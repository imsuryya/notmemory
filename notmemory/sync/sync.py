# notmemory/sync/sync.py
"""
Multi-agent memory sync for notmemory.

Synchronises memory banks between multiple AgentMemory instances,
respecting the permission store — only entries the source agent
is allowed to read are synced to the target agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from notmemory.core.agent_memory import AgentMemory
from notmemory.sync.permissions import AccessLevel, PermissionStore


@dataclass
class SyncConfig:
    """Configuration for a MemorySync operation."""

    overwrite_existing: bool = False
    sync_tombstoned: bool = False
    limit: int = 500


@dataclass
class SyncResult:
    """Result of a sync operation."""

    bank_id: str
    source_agent_id: str
    target_agent_id: str
    entries_synced: int = 0
    entries_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "entries_synced": self.entries_synced,
            "entries_skipped": self.entries_skipped,
            "errors": self.errors,
            "success": self.success,
        }


class MemorySync:
    """
    Synchronises memory banks between AgentMemory instances.

    Respects the PermissionStore — source agent must have READ,
    target agent must have WRITE.
    """

    def __init__(
        self,
        permissions: PermissionStore | None = None,
        config: SyncConfig | None = None,
    ) -> None:
        self._permissions = permissions or PermissionStore()
        self._config = config or SyncConfig()

    async def sync_bank(
        self,
        *,
        source: AgentMemory,
        target: AgentMemory,
        bank_id: str,
        source_agent_id: str,
        target_agent_id: str,
    ) -> SyncResult:
        """
        Sync all entries from source bank to target bank.

        Permission checks:
          - source_agent_id must have READ on bank_id
          - target_agent_id must have WRITE on bank_id
        """
        result = SyncResult(
            bank_id=bank_id,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
        )

        # Permission checks
        if not self._permissions.check(source_agent_id, bank_id, AccessLevel.READ):
            result.errors.append(f"Agent '{source_agent_id}' lacks READ on bank '{bank_id}'")
            return result

        if not self._permissions.check(target_agent_id, bank_id, AccessLevel.WRITE):
            result.errors.append(f"Agent '{target_agent_id}' lacks WRITE on bank '{bank_id}'")
            return result

        # Read from source
        try:
            recall = await source.recall(
                bank_id=bank_id,
                limit=self._config.limit,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Source recall failed: {exc}")
            return result

        # Get existing target content fingerprints to detect duplicates
        existing_fingerprints: set[str] = set()
        if not self._config.overwrite_existing:
            try:
                existing = await target.recall(
                    bank_id=bank_id,
                    limit=self._config.limit,
                )
                existing_fingerprints = {
                    json.dumps(e.content, sort_keys=True) for e in existing.entries
                }
            except Exception:  # noqa: BLE001
                pass

        # Write to target
        for entry in recall.entries:
            if entry.is_tombstoned and not self._config.sync_tombstoned:
                result.entries_skipped += 1
                continue

            fingerprint = json.dumps(entry.content, sort_keys=True)
            if fingerprint in existing_fingerprints:
                result.entries_skipped += 1
                continue

            try:
                await target.retain(
                    bank_id=bank_id,
                    content=entry.content,
                    context=entry.context,
                    source=entry.source,
                    trust_level=entry.trust_level,
                    confidence=entry.confidence,
                )
                result.entries_synced += 1
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"Failed to sync entry {entry.id}: {exc}")
                result.entries_skipped += 1

        return result

    async def sync_all_banks(
        self,
        *,
        source: AgentMemory,
        target: AgentMemory,
        bank_ids: list[str],
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[SyncResult]:
        """Sync multiple banks at once. Returns one SyncResult per bank."""
        results = []
        for bank_id in bank_ids:
            result = await self.sync_bank(
                source=source,
                target=target,
                bank_id=bank_id,
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
            )
            results.append(result)
        return results


__all__ = ["MemorySync", "SyncConfig", "SyncResult"]