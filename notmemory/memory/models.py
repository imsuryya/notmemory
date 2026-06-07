from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

TrustLevel = Literal["hypothesis", "active", "validated", "deprecated"]
RecallStrategy = Literal["semantic", "keyword", "graph", "temporal", "hybrid"]
CycleEventType = Literal[
    "llm-call", "tool-execution", "gate-decision", "memory-write", "memory-read"
]
ConflictType = Literal["contradiction", "staleness", "duplicate", "schema_mismatch"]


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"mem-{uuid.uuid4().hex[:12]}")
    bank_id: str
    content: dict[str, Any]
    context: str | None = None
    source: str | None = None
    trust_level: TrustLevel = "active"
    confidence: float = 1.0
    transaction_id: str = Field(default_factory=lambda: f"txn-{uuid.uuid4().hex[:12]}")
    hash: str | None = None
    parent_hash: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_tombstoned: bool = False


class RecallResult(BaseModel):
    bank_id: str
    query: str
    strategies_used: list[RecallStrategy]
    entries: list[MemoryEntry]
    token_count: int = 0
    elapsed_ms: float = 0.0


class Conflict(BaseModel):
    conflict_type: ConflictType
    entry_a_id: str
    entry_b_id: str
    description: str
    severity: float


class ConflictReport(BaseModel):
    bank_id: str
    health_score: float
    conflicts: list[Conflict] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CycleEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    cycle_id: str
    parent_event_id: str | None = None
    event_type: CycleEventType
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: float = 0.0
    output: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AuditTrail(BaseModel):
    cycle_id: str
    events: list[CycleEvent]
    total_tokens: int = 0
    total_cost_usd: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class RollbackResult(BaseModel):
    transaction_id: str
    entries_reversed: int
    rolled_back_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    success: bool = True
