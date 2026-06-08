# notmemory/adapters/mem0.py
"""
Mem0 sync-layer adapter for notmemory.

Every retain() writes to SQLite (hash chain intact) AND mirrors to Mem0.
Mem0 becomes a semantic search sidecar — notmemory stays the source of truth.

Install:
    pip install -e ".[mem0]"

Usage:
    from notmemory.adapters.mem0 import NotMemoryMem0Adapter

    adapter = NotMemoryMem0Adapter(
        mem0_api_key="your-key",   # or set MEM0_API_KEY env var
        user_id="agent-1",
    )
    async with adapter:
        entry = await adapter.retain(
            bank_id="facts",
            content={"fact": "Paris is the capital of France"},
        )
        results = await adapter.semantic_recall("capital of France")
"""

from __future__ import annotations

import os
from typing import Any

from notmemory.adapters.base import BaseAdapter
from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig
from notmemory.memory.models import MemoryEntry, RecallResult, TrustLevel

try:
    from mem0 import AsyncMemoryClient
except ImportError as exc:  # pragma: no cover
    raise ImportError("Mem0 extras are required. Install with: pip install -e '.[mem0]'") from exc


class NotMemoryMem0Adapter(BaseAdapter):
    """
    Sync-layer adapter that mirrors notmemory entries to Mem0.

    Architecture:
      - SQLite (via AgentMemory) = source of truth, hash chain, audit trail
      - Mem0 = semantic search sidecar

    Every ``retain()`` call:
      1. Writes to SQLite with full hash chaining
      2. Mirrors the content to Mem0 for semantic search

    ``semantic_recall()`` queries Mem0 directly for vector similarity search.
    ``recall()`` queries SQLite as normal (keyword/temporal).
    """

    @property
    def adapter_name(self) -> str:
        return "mem0"

    @property
    def adapter_version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        mem0_api_key: str | None = None,
        user_id: str = "default",
        memory: AgentMemory | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """
        Args:
            mem0_api_key: Mem0 API key. Falls back to MEM0_API_KEY env var.
            user_id:      Mem0 user/agent identifier for namespacing memories.
            memory:       Existing AgentMemory instance (optional).
            config:       MemoryConfig for a fresh AgentMemory (optional).
        """
        super().__init__(memory=memory, config=config)
        self._api_key = mem0_api_key or os.environ.get("MEM0_API_KEY", "")
        self._user_id = user_id
        self._mem0: AsyncMemoryClient | None = None

    # ── lifecycle ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize AgentMemory and Mem0 client."""
        await self._memory.initialize()
        self._mem0 = AsyncMemoryClient(api_key=self._api_key)
        self._initialized = True

    async def teardown(self) -> None:
        """Close AgentMemory. Mem0 client is stateless — no close needed."""
        await self._memory.close()
        self._mem0 = None
        self._initialized = False

    async def __aenter__(self) -> NotMemoryMem0Adapter:
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.teardown()

    # ── core operations ────────────────────────────────────────────────

    async def retain(
        self,
        *,
        bank_id: str,
        content: dict[str, Any],
        context: str | None = None,
        source: str | None = None,
        trust_level: TrustLevel = "active",
        confidence: float = 1.0,
    ) -> MemoryEntry:
        """
        Store in SQLite (hash chain) AND mirror to Mem0 (semantic index).

        Returns the notmemory MemoryEntry — SQLite is the source of truth.
        """
        self._check_initialized()

        # 1. Write to SQLite with full hash chaining
        entry = await self._memory.retain(
            bank_id=bank_id,
            content=content,
            context=context,
            source=source,
            trust_level=trust_level,
            confidence=confidence,
        )

        # 2. Mirror to Mem0 — best effort, don't fail if Mem0 is down
        if self._mem0 is not None:
            try:
                text = _content_to_text(content)
                await self._mem0.add(
                    messages=[{"role": "user", "content": text}],
                    user_id=self._user_id,
                    metadata={
                        "notmemory_id": entry.id,
                        "bank_id": bank_id,
                        "transaction_id": entry.transaction_id,
                    },
                )
            except Exception:  # noqa: BLE001
                # Mem0 mirror is best-effort — SQLite write already succeeded
                pass

        return entry

    async def recall(
        self,
        *,
        bank_id: str,
        query: str | None = None,
        limit: int = 20,
    ) -> RecallResult:
        """
        Keyword/temporal recall from SQLite (standard notmemory recall).
        For semantic search use ``semantic_recall()``.
        """
        self._check_initialized()
        return await self._memory.recall(
            bank_id=bank_id,
            query=query,
            limit=limit,
        )

    async def semantic_recall(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Semantic vector search via Mem0.

        Returns Mem0's raw result dicts (include ``id``, ``memory``,
        ``score``, and ``metadata`` with ``notmemory_id`` for cross-reference).
        """
        self._check_initialized()
        if self._mem0 is None:
            return []
        results = await self._mem0.search(
            query=query,
            user_id=self._user_id,
            limit=limit,
        )
        return results if isinstance(results, list) else []

    async def rollback(self, transaction_id: str) -> Any:
        """
        Rollback in SQLite (tombstone). Does NOT delete from Mem0 —
        the hash chain records the rollback; Mem0 is a search index only.
        """
        self._check_initialized()
        return await self._memory.rollback(transaction_id)

    async def forget(
        self,
        bank_id: str,
        *,
        entry_ids: list[str] | None = None,
    ) -> int:
        """
        GDPR tombstone in SQLite. Also attempts to delete from Mem0
        if entry_ids are provided (best effort).
        """
        self._check_initialized()

        # Tombstone in SQLite
        count = await self._memory.forget(bank_id, entry_ids=entry_ids)

        # Best-effort delete from Mem0 by notmemory_id
        if self._mem0 is not None and entry_ids:
            for eid in entry_ids:
                try:
                    # Search for the Mem0 memory with this notmemory_id
                    results = await self._mem0.search(
                        query=eid,
                        user_id=self._user_id,
                        limit=5,
                    )
                    for r in results or []:
                        meta = r.get("metadata", {})
                        if meta.get("notmemory_id") == eid:
                            await self._mem0.delete(r["id"])
                except Exception:  # noqa: BLE001
                    pass

        return count

    # ── introspection ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "user_id": self._user_id,
            "mem0_connected": self._mem0 is not None,
            "api_key_set": bool(self._api_key),
        }

    # ── internal ───────────────────────────────────────────────────────

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("NotMemoryMem0Adapter not initialized. Use: async with adapter: ...")


def _content_to_text(content: dict[str, Any]) -> str:
    """Convert a content dict to a plain text string for Mem0 indexing."""
    parts = []
    for k, v in content.items():
        if isinstance(v, str):
            parts.append(v)
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else str(content)


__all__ = ["NotMemoryMem0Adapter"]
