import pytest
from pydantic import BaseModel

from notmemory import AgentMemory, MemoryConfig
from notmemory.core.exceptions import RollbackError, ValidationError


@pytest.fixture
async def memory():
    async with AgentMemory(MemoryConfig(db_url="sqlite+aiosqlite:///:memory:")) as m:
        yield m


@pytest.mark.asyncio
async def test_retain_and_recall(memory):
    entry = await memory.retain(bank_id="test", content={"name": "Alice"})
    assert entry.id.startswith("mem-")
    assert entry.hash is not None
    result = await memory.recall(bank_id="test")
    assert len(result.entries) == 1
    assert result.entries[0].content["name"] == "Alice"


@pytest.mark.asyncio
async def test_schema_validation_fails(memory):
    class User(BaseModel):
        name: str
        age: int

    with pytest.raises(ValidationError):
        await memory.retain(
            bank_id="users",
            content={"name": "Alice", "age": "not-an-int"},
            schema=User,
        )


@pytest.mark.asyncio
async def test_rollback(memory):
    entry = await memory.retain(bank_id="facts", content={"claim": "sky is green"})
    result = await memory.rollback(entry.transaction_id)
    assert result.success
    assert result.entries_reversed == 1
    recall = await memory.recall(bank_id="facts")
    assert all(e.id != entry.id for e in recall.entries)


@pytest.mark.asyncio
async def test_rollback_nonexistent_raises(memory):
    with pytest.raises(RollbackError):
        await memory.rollback("txn-does-not-exist")


@pytest.mark.asyncio
async def test_hash_chain_intact(memory):
    await memory.retain(bank_id="chain", content={"seq": 1})
    await memory.retain(bank_id="chain", content={"seq": 2})
    assert await memory.verify_integrity("chain") is True


@pytest.mark.asyncio
async def test_conflict_duplicate(memory):
    await memory.retain(bank_id="facts", content={"x": 1})
    await memory.retain(bank_id="facts", content={"x": 1})
    report = await memory.detect_conflicts(bank_id="facts")
    assert report.health_score < 100
    assert len(report.conflicts) == 1
    assert report.conflicts[0].conflict_type == "duplicate"


@pytest.mark.asyncio
async def test_no_conflicts(memory):
    await memory.retain(bank_id="clean", content={"a": 1})
    await memory.retain(bank_id="clean", content={"b": 2})
    report = await memory.detect_conflicts(bank_id="clean")
    assert report.health_score == 100.0


@pytest.mark.asyncio
async def test_audit_trail(memory):
    evt = await memory.log_cycle_event(
        cycle_id="cyc-001",
        event_type="llm-call",
        model="gpt-4o",
        tokens_in=100,
        tokens_out=200,
        elapsed_ms=450.0,
    )
    await memory.log_cycle_event(
        cycle_id="cyc-001",
        parent_event_id=evt.event_id,
        event_type="tool-execution",
        elapsed_ms=120.0,
    )
    trail = await memory.get_audit_trail(cycle_id="cyc-001")
    assert len(trail.events) == 2
    assert trail.total_tokens == 300


@pytest.mark.asyncio
async def test_forget(memory):
    await memory.retain(bank_id="gdpr", content={"pii": "user@example.com"})
    await memory.retain(bank_id="gdpr", content={"pii": "555-1234"})
    count = await memory.forget("gdpr")
    assert count == 2
    recall = await memory.recall(bank_id="gdpr")
    assert recall.entries == []


@pytest.mark.asyncio
async def test_use_before_init_raises():
    memory = AgentMemory(MemoryConfig(db_url="sqlite+aiosqlite:///:memory:"))
    with pytest.raises(RuntimeError, match="not initialized"):
        await memory.retain(bank_id="x", content={"x": 1})
