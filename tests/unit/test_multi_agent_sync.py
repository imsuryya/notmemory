# tests/unit/test_multi_agent_sync.py
"""Tests for multi-agent sync and permissions."""

from __future__ import annotations

import uuid

import pytest

from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig
from notmemory.sync.permissions import AccessLevel, AgentPermission, PermissionStore
from notmemory.sync.sync import MemorySync, SyncResult


def _config() -> MemoryConfig:
    unique = uuid.uuid4().hex
    return MemoryConfig(
        db_url=(f"sqlite+aiosqlite:///file:{unique}?mode=memory&cache=shared&uri=true")
    )


async def _memory() -> AgentMemory:
    m = AgentMemory(config=_config())
    await m.initialize()
    return m


# ── AccessLevel ───────────────────────────────────────────────────────────────


def test_access_level_ordering():
    assert AccessLevel.ADMIN.allows(AccessLevel.READ)
    assert AccessLevel.ADMIN.allows(AccessLevel.WRITE)
    assert AccessLevel.ADMIN.allows(AccessLevel.ADMIN)
    assert AccessLevel.WRITE.allows(AccessLevel.READ)
    assert AccessLevel.WRITE.allows(AccessLevel.WRITE)
    assert not AccessLevel.WRITE.allows(AccessLevel.ADMIN)
    assert AccessLevel.READ.allows(AccessLevel.READ)
    assert not AccessLevel.READ.allows(AccessLevel.WRITE)
    assert not AccessLevel.NONE.allows(AccessLevel.READ)


def test_access_level_values():
    assert AccessLevel.READ.value == "read"
    assert AccessLevel.WRITE.value == "write"
    assert AccessLevel.ADMIN.value == "admin"
    assert AccessLevel.NONE.value == "none"


# ── AgentPermission ───────────────────────────────────────────────────────────


def test_agent_permission_can_read():
    p = AgentPermission("a1", "facts", AccessLevel.READ)
    assert p.can_read()
    assert not p.can_write()
    assert not p.can_admin()


def test_agent_permission_can_write():
    p = AgentPermission("a1", "facts", AccessLevel.WRITE)
    assert p.can_read()
    assert p.can_write()
    assert not p.can_admin()


def test_agent_permission_can_admin():
    p = AgentPermission("a1", "facts", AccessLevel.ADMIN)
    assert p.can_read()
    assert p.can_write()
    assert p.can_admin()


def test_agent_permission_to_dict():
    p = AgentPermission("a1", "facts", AccessLevel.WRITE)
    d = p.to_dict()
    assert d["agent_id"] == "a1"
    assert d["bank_id"] == "facts"
    assert d["access_level"] == "write"


# ── PermissionStore ───────────────────────────────────────────────────────────


def test_grant_and_check():
    store = PermissionStore()
    store.grant("agent-1", "facts", AccessLevel.WRITE)
    assert store.check("agent-1", "facts", AccessLevel.READ)
    assert store.check("agent-1", "facts", AccessLevel.WRITE)
    assert not store.check("agent-1", "facts", AccessLevel.ADMIN)


def test_revoke():
    store = PermissionStore()
    store.grant("agent-1", "facts", AccessLevel.WRITE)
    store.revoke("agent-1", "facts")
    assert not store.check("agent-1", "facts", AccessLevel.READ)


def test_wildcard_permission():
    store = PermissionStore()
    store.grant("agent-1", "*", AccessLevel.READ)
    assert store.check("agent-1", "facts", AccessLevel.READ)
    assert store.check("agent-1", "other-bank", AccessLevel.READ)
    assert not store.check("agent-1", "facts", AccessLevel.WRITE)


def test_specific_overrides_wildcard():
    store = PermissionStore()
    store.grant("agent-1", "*", AccessLevel.READ)
    store.grant("agent-1", "facts", AccessLevel.ADMIN)
    assert store.check("agent-1", "facts", AccessLevel.ADMIN)
    assert not store.check("agent-1", "other", AccessLevel.ADMIN)


def test_no_permission_returns_false():
    store = PermissionStore()
    assert not store.check("agent-x", "facts", AccessLevel.READ)


def test_list_permissions():
    store = PermissionStore()
    store.grant("agent-1", "facts", AccessLevel.READ)
    store.grant("agent-1", "logs", AccessLevel.WRITE)
    store.grant("agent-2", "facts", AccessLevel.ADMIN)

    all_perms = store.list_permissions()
    assert len(all_perms) == 3

    agent1_perms = store.list_permissions(agent_id="agent-1")
    assert len(agent1_perms) == 2


def test_all_agents():
    store = PermissionStore()
    store.grant("agent-1", "facts", AccessLevel.READ)
    store.grant("agent-2", "facts", AccessLevel.WRITE)
    agents = store.all_agents()
    assert "agent-1" in agents
    assert "agent-2" in agents


# ── MemorySync ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_requires_source_read_permission():
    store = PermissionStore()
    store.grant("agent-b", "facts", AccessLevel.WRITE)
    # agent-a has no READ

    sync = MemorySync(permissions=store)
    source = await _memory()
    target = await _memory()

    result = await sync.sync_bank(
        source=source,
        target=target,
        bank_id="facts",
        source_agent_id="agent-a",
        target_agent_id="agent-b",
    )
    assert not result.success
    assert "lacks READ" in result.errors[0]

    await source.close()
    await target.close()


@pytest.mark.asyncio
async def test_sync_requires_target_write_permission():
    store = PermissionStore()
    store.grant("agent-a", "facts", AccessLevel.READ)
    # agent-b has no WRITE

    sync = MemorySync(permissions=store)
    source = await _memory()
    target = await _memory()

    result = await sync.sync_bank(
        source=source,
        target=target,
        bank_id="facts",
        source_agent_id="agent-a",
        target_agent_id="agent-b",
    )
    assert not result.success
    assert "lacks WRITE" in result.errors[0]

    await source.close()
    await target.close()


@pytest.mark.asyncio
async def test_sync_copies_entries():
    store = PermissionStore()
    store.grant("agent-a", "facts", AccessLevel.READ)
    store.grant("agent-b", "facts", AccessLevel.WRITE)

    sync = MemorySync(permissions=store)
    source = await _memory()
    target = await _memory()

    await source.retain(
        bank_id="facts",
        content={"fact": "sky is blue"},
    )
    await source.retain(
        bank_id="facts",
        content={"fact": "water is wet"},
    )

    result = await sync.sync_bank(
        source=source,
        target=target,
        bank_id="facts",
        source_agent_id="agent-a",
        target_agent_id="agent-b",
    )

    assert result.success
    assert result.entries_synced == 2
    assert result.entries_skipped == 0

    target_recall = await target.recall(bank_id="facts")
    assert len(target_recall.entries) == 2

    await source.close()
    await target.close()


@pytest.mark.asyncio
async def test_sync_skips_duplicates():
    store = PermissionStore()
    store.grant("agent-a", "facts", AccessLevel.ADMIN)
    store.grant("agent-b", "facts", AccessLevel.ADMIN)
    store.grant("agent-c", "facts", AccessLevel.ADMIN)

    sync = MemorySync(permissions=store)
    source = await _memory()
    target = await _memory()
    third = await _memory()

    await source.retain(bank_id="facts", content={"fact": "sky is blue"})

    # Sync source → target (1 entry)
    result1 = await sync.sync_bank(
        source=source,
        target=target,
        bank_id="facts",
        source_agent_id="agent-a",
        target_agent_id="agent-b",
    )
    assert result1.entries_synced == 1

    # Pre-populate third with same entry IDs as target
    target_recall = await target.recall(bank_id="facts")
    for e in target_recall.entries:
        await third.retain(
            bank_id="facts",
            content=e.content,
            context=e.context,
            source=e.source,
        )

    # Sync target → third — third already has matching IDs, should skip
    result2 = await sync.sync_bank(
        source=target,
        target=third,
        bank_id="facts",
        source_agent_id="agent-b",
        target_agent_id="agent-c",
    )
    assert result2.entries_skipped == 1
    assert result2.entries_synced == 0

    await source.close()
    await target.close()
    await third.close()


@pytest.mark.asyncio
async def test_sync_all_banks():
    store = PermissionStore()
    store.grant("agent-a", "*", AccessLevel.READ)
    store.grant("agent-b", "*", AccessLevel.WRITE)

    sync = MemorySync(permissions=store)
    source = await _memory()
    target = await _memory()

    await source.retain(bank_id="facts", content={"f": "1"})
    await source.retain(bank_id="logs", content={"l": "1"})

    results = await sync.sync_all_banks(
        source=source,
        target=target,
        bank_ids=["facts", "logs"],
        source_agent_id="agent-a",
        target_agent_id="agent-b",
    )

    assert len(results) == 2
    assert all(r.success for r in results)
    assert sum(r.entries_synced for r in results) == 2

    await source.close()
    await target.close()


@pytest.mark.asyncio
async def test_sync_result_to_dict():
    result = SyncResult(
        bank_id="facts",
        source_agent_id="agent-a",
        target_agent_id="agent-b",
        entries_synced=3,
        entries_skipped=1,
    )
    d = result.to_dict()
    assert d["entries_synced"] == 3
    assert d["success"] is True
    assert d["bank_id"] == "facts"
