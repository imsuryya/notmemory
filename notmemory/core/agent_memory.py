from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from notmemory.core.config import MemoryConfig
from notmemory.core.exceptions import ValidationError
from notmemory.memory.models import (
    AuditTrail,
    ConflictReport,
    CycleEvent,
    CycleEventType,
    MemoryEntry,
    RecallResult,
    RecallStrategy,
    RollbackResult,
    TrustLevel,
)
from notmemory.storage import create_backend
from notmemory.storage.backends.base import BaseStorageBackend


class AgentMemory:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()
        self._backend: BaseStorageBackend = create_backend(self._config)
        self._initialized = False

    async def initialize(self) -> None:
        await self._backend.initialize()
        self._initialized = True

    async def close(self) -> None:
        await self._backend.close()
        self._initialized = False

    async def __aenter__(self) -> AgentMemory:
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _check(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "AgentMemory not initialized. "
                "Call await memory.initialize() or use: async with AgentMemory() as memory:"
            )

    async def retain(
        self,
        *,
        bank_id: str,
        content: dict[str, Any] | BaseModel,
        context: str | None = None,
        source: str | None = None,
        schema: type[BaseModel] | None = None,
        trust_level: TrustLevel = "active",
        confidence: float = 1.0,
    ) -> MemoryEntry:
        self._check()

        if isinstance(content, BaseModel):
            raw = content.model_dump()
        else:
            raw = content

        if schema is not None:
            try:
                schema.model_validate(raw)
            except Exception as exc:
                raise ValidationError(
                    f"Schema validation failed for bank '{bank_id}': {exc}"
                ) from exc

        entry = MemoryEntry(
            bank_id=bank_id,
            content=raw,
            context=context,
            source=source,
            trust_level=trust_level,
            confidence=confidence,
        )
        return await self._backend.write_entry(entry)

    async def recall(
        self,
        *,
        bank_id: str,
        query: str | None = None,
        strategies: list[RecallStrategy] | None = None,
        trust_level: TrustLevel | None = None,
        token_limit: int = 2000,
        limit: int = 20,
    ) -> RecallResult:
        self._check()

        t0 = time.monotonic()

        if strategies is None:
            strategies = ["keyword"] if query else ["temporal"]

        entries: list[MemoryEntry] = []
        seen_ids: set[str] = set()

        for strategy in strategies:
            if strategy in ("keyword", "hybrid") and query:
                found = await self._backend.keyword_search(bank_id, query, limit=limit)
            else:
                found = await self._backend.read_entries(
                    bank_id, trust_level=trust_level, limit=limit
                )
            for e in found:
                if e.id not in seen_ids:
                    entries.append(e)
                    seen_ids.add(e.id)

        total_tokens = 0
        filtered: list[MemoryEntry] = []
        for entry in entries:
            est = len(str(entry.content)) // 4
            if total_tokens + est > token_limit:
                break
            filtered.append(entry)
            total_tokens += est

        return RecallResult(
            bank_id=bank_id,
            query=query or "",
            strategies_used=strategies,
            entries=filtered,
            token_count=total_tokens,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    async def rollback(self, transaction_id: str) -> RollbackResult:
        self._check()
        return await self._backend.rollback_transaction(transaction_id)

    async def detect_conflicts(
        self,
        *,
        bank_id: str,
        trust_level: TrustLevel | None = None,
    ) -> ConflictReport:
        self._check()
        return await self._backend.detect_conflicts(bank_id, trust_level=trust_level)

    async def log_cycle_event(
        self,
        *,
        cycle_id: str,
        event_type: CycleEventType,
        parent_event_id: str | None = None,
        model: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        elapsed_ms: float = 0.0,
        output: dict[str, Any] | None = None,
    ) -> CycleEvent:
        self._check()
        event = CycleEvent(
            cycle_id=cycle_id,
            parent_event_id=parent_event_id,
            event_type=event_type,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            elapsed_ms=elapsed_ms,
            output=output,
        )
        return await self._backend.write_cycle_event(event)

    async def get_audit_trail(
        self,
        *,
        cycle_id: str,
        include: list[CycleEventType] | None = None,
    ) -> AuditTrail:
        self._check()
        return await self._backend.get_audit_trail(cycle_id, event_types=include)

    async def verify_integrity(self, bank_id: str) -> bool:
        self._check()
        return await self._backend.verify_hash_chain(bank_id)

    async def forget(
        self,
        bank_id: str,
        *,
        entry_ids: list[str] | None = None,
    ) -> int:
        self._check()
        return await self._backend.tombstone_entries(bank_id, entry_ids=entry_ids)
