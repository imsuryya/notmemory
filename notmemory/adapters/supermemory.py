# notmemory/adapters/supermemory.py
"""
SuperMemory sync-layer adapter for notmemory.

Every retain() writes to SQLite (hash chain intact) AND mirrors to SuperMemory.
SuperMemory becomes a semantic search sidecar — notmemory stays source of truth.

Install:
    pip install httpx  (already a transitive dep)

Usage:
    from notmemory.adapters.supermemory import NotMemorySuperMemoryAdapter

    adapter = NotMemorySuperMemoryAdapter(
        api_key="sm_...",   # or set SUPERMEMORY_API_KEY env var
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

import httpx

from notmemory.adapters.base import BaseAdapter
from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig
from notmemory.memory.models import MemoryEntry, RecallResult, TrustLevel

SUPERMEMORY_BASE_URL = "https://api.supermemory.ai/v3"


class NotMemorySuperMemoryAdapter(BaseAdapter):
    """
    Sync-layer adapter that mirrors notmemory entries to SuperMemory.

    Architecture:
      - SQLite (via AgentMemory) = source of truth, hash chain, audit trail
      - SuperMemory = semantic search sidecar

    Every ``retain()`` call:
      1. Writes to SQLite with full hash chaining
      2. Mirrors the content to SuperMemory for semantic search

    ``semantic_recall()`` queries SuperMemory for vector similarity search.
    ``recall()`` queries SQLite as normal (keyword/temporal).
    """

    @property
    def adapter_name(self) -> str:
        return "supermemory"

    @property
    def adapter_version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str = "default",
        base_url: str = SUPERMEMORY_BASE_URL,
        memory: AgentMemory | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """
        Args:
            api_key:  SuperMemory API key. Falls back to
                      SUPERMEMORY_API_KEY env var.
            user_id:  SuperMemory container/user id for namespacing.
            base_url: SuperMemory API base URL (override for testing).
            memory:   Existing AgentMemory instance (optional).
            config:   MemoryConfig for a fresh AgentMemory (optional).
        """
        super().__init__(memory=memory, config=config)
        self._api_key = api_key or os.environ.get("SUPERMEMORY_API_KEY", "")
        self._user_id = user_id
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # ── lifecycle ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        await self._memory.initialize()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._initialized = True

    async def teardown(self) -> None:
        await self._memory.close()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    async def __aenter__(self) -> NotMemorySuperMemoryAdapter:
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
        Store in SQLite (hash chain) AND mirror to SuperMemory.
        Returns the notmemory MemoryEntry — SQLite is source of truth.
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

        # 2. Mirror to SuperMemory — best effort
        if self._client is not None:
            try:
                await self._client.post(
                    "/memories",
                    json={
                        "content": _content_to_text(content),
                        "userId": self._user_id,
                        "metadata": {
                            "notmemory_id": entry.id,
                            "bank_id": bank_id,
                            "transaction_id": entry.transaction_id,
                        },
                    },
                )
            except Exception:  # noqa: BLE001
                # Mirror is best-effort — SQLite write already succeeded
                pass

        return entry

    async def recall(
        self,
        *,
        bank_id: str,
        query: str | None = None,
        limit: int = 20,
    ) -> RecallResult:
        """Keyword/temporal recall from SQLite."""
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
        Semantic vector search via SuperMemory.

        Returns list of dicts with ``id``, ``content``, ``score``,
        and ``metadata`` (includes ``notmemory_id`` for cross-reference).
        """
        self._check_initialized()
        if self._client is None:
            return []
        try:
            response = await self._client.get(
                "/memories/search",
                params={
                    "q": query,
                    "userId": self._user_id,
                    "limit": limit,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("results", [])
        except Exception:  # noqa: BLE001
            return []

    async def rollback(self, transaction_id: str) -> Any:
        """Tombstone in SQLite. SuperMemory is a search index only."""
        self._check_initialized()
        return await self._memory.rollback(transaction_id)

    async def forget(
        self,
        bank_id: str,
        *,
        entry_ids: list[str] | None = None,
    ) -> int:
        """
        GDPR tombstone in SQLite + best-effort delete from SuperMemory.
        """
        self._check_initialized()

        count = await self._memory.forget(bank_id, entry_ids=entry_ids)

        if self._client is not None and entry_ids:
            for eid in entry_ids:
                try:
                    response = await self._client.get(
                        "/memories/search",
                        params={"q": eid, "userId": self._user_id, "limit": 5},
                    )
                    data = response.json()
                    results = data if isinstance(data, list) else data.get("results", [])
                    for r in results:
                        meta = r.get("metadata", {})
                        if meta.get("notmemory_id") == eid:
                            await self._client.delete(f"/memories/{r['id']}")
                except Exception:  # noqa: BLE001
                    pass

        return count

    # ── introspection ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "user_id": self._user_id,
            "base_url": self._base_url,
            "client_open": self._client is not None,
            "api_key_set": bool(self._api_key),
        }

    # ── internal ───────────────────────────────────────────────────────

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "NotMemorySuperMemoryAdapter not initialized. Use: async with adapter: ..."
            )


def _content_to_text(content: dict[str, Any]) -> str:
    """Convert content dict to plain text for SuperMemory indexing."""
    parts = []
    for k, v in content.items():
        if isinstance(v, str):
            parts.append(v)
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else str(content)


__all__ = ["NotMemorySuperMemoryAdapter"]
