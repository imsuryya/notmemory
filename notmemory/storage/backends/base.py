from __future__ import annotations
from abc import ABC, abstractmethod
from notmemory.memory.models import (
    AuditTrail, ConflictReport, CycleEvent,
    MemoryEntry, RollbackResult, TrustLevel,
)


class BaseStorageBackend(ABC):

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def write_entry(self, entry: MemoryEntry) -> MemoryEntry:
        pass

    @abstractmethod
    async def read_entries(
        self,
        bank_id: str,
        *,
        trust_level: TrustLevel | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        pass

    @abstractmethod
    async def keyword_search(
        self,
        bank_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        pass

    @abstractmethod
    async def rollback_transaction(self, transaction_id: str) -> RollbackResult:
        pass

    @abstractmethod
    async def write_cycle_event(self, event: CycleEvent) -> CycleEvent:
        pass

    @abstractmethod
    async def get_audit_trail(
        self,
        cycle_id: str,
        *,
        event_types: list[str] | None = None,
    ) -> AuditTrail:
        pass

    @abstractmethod
    async def detect_conflicts(
        self,
        bank_id: str,
        *,
        trust_level: TrustLevel | None = None,
    ) -> ConflictReport:
        pass

    @abstractmethod
    async def verify_hash_chain(self, bank_id: str) -> bool:
        pass

    @abstractmethod
    async def tombstone_entries(
        self,
        bank_id: str,
        *,
        entry_ids: list[str] | None = None,
    ) -> int:
        pass