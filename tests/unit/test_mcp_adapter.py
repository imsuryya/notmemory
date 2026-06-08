# tests/unit/test_mcp_adapter.py
"""Tests for the MCP server adapter."""

from __future__ import annotations

import uuid

import pytest

from notmemory.adapters.mcp import TOOLS, _dispatch, create_server
from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig


def _config() -> MemoryConfig:
    unique = uuid.uuid4().hex
    return MemoryConfig(
        db_url=(f"sqlite+aiosqlite:///file:{unique}?mode=memory&cache=shared&uri=true")
    )


async def _memory(config: MemoryConfig) -> AgentMemory:
    m = AgentMemory(config=config)
    await m.initialize()
    return m


# ── tool registry ─────────────────────────────────────────────────────────────


def test_tools_registered():
    names = {t.name for t in TOOLS}
    assert "notmemory_retain" in names
    assert "notmemory_recall" in names
    assert "notmemory_rollback" in names
    assert "notmemory_forget" in names
    assert "notmemory_health" in names


def test_all_tools_have_schemas():
    for tool in TOOLS:
        assert tool.inputSchema is not None
        assert "type" in tool.inputSchema


def test_create_server_returns_server():
    server = create_server(config=_config())
    assert server is not None
    assert server.name == "notmemory"


# ── retain tool ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retain_returns_entry_metadata():
    m = await _memory(_config())
    result = await _dispatch(
        m,
        "notmemory_retain",
        {
            "bank_id": "facts",
            "content": {"fact": "sky is blue"},
        },
    )
    assert "id" in result
    assert "transaction_id" in result
    assert result["bank_id"] == "facts"
    assert result["confidence"] == 1.0
    await m.close()


@pytest.mark.asyncio
async def test_retain_with_optional_fields():
    m = await _memory(_config())
    result = await _dispatch(
        m,
        "notmemory_retain",
        {
            "bank_id": "facts",
            "content": {"fact": "water is wet"},
            "context": "science",
            "source": "agent",
            "confidence": 0.9,
        },
    )
    assert result["confidence"] == 0.9
    await m.close()


# ── recall tool ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_empty_bank():
    m = await _memory(_config())
    result = await _dispatch(m, "notmemory_recall", {"bank_id": "facts"})
    assert result["entries"] == []
    assert result["bank_id"] == "facts"
    await m.close()


@pytest.mark.asyncio
async def test_recall_returns_stored_entries():
    m = await _memory(_config())
    await _dispatch(
        m,
        "notmemory_retain",
        {
            "bank_id": "facts",
            "content": {"fact": "gravity exists"},
        },
    )
    result = await _dispatch(
        m,
        "notmemory_recall",
        {
            "bank_id": "facts",
            "limit": 5,
        },
    )
    assert len(result["entries"]) == 1
    assert result["entries"][0]["content"]["fact"] == "gravity exists"
    await m.close()


# ── rollback tool ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rollback_tombstones_transaction():
    m = await _memory(_config())
    retained = await _dispatch(
        m,
        "notmemory_retain",
        {
            "bank_id": "facts",
            "content": {"fact": "bad fact"},
        },
    )
    result = await _dispatch(
        m,
        "notmemory_rollback",
        {
            "transaction_id": retained["transaction_id"],
        },
    )
    assert result["success"] is True
    assert result["entries_reversed"] == 1

    # Verify entry is gone
    recalled = await _dispatch(m, "notmemory_recall", {"bank_id": "facts"})
    assert recalled["entries"] == []
    await m.close()


# ── forget tool ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_entire_bank():
    m = await _memory(_config())
    await _dispatch(
        m,
        "notmemory_retain",
        {
            "bank_id": "temp",
            "content": {"data": "delete me"},
        },
    )
    result = await _dispatch(m, "notmemory_forget", {"bank_id": "temp"})
    assert result["entries_tombstoned"] == 1
    assert result["bank_id"] == "temp"
    await m.close()


@pytest.mark.asyncio
async def test_forget_specific_entry():
    m = await _memory(_config())
    retained = await _dispatch(
        m,
        "notmemory_retain",
        {
            "bank_id": "temp",
            "content": {"data": "specific"},
        },
    )
    result = await _dispatch(
        m,
        "notmemory_forget",
        {
            "bank_id": "temp",
            "entry_ids": [retained["id"]],
        },
    )
    assert result["entries_tombstoned"] == 1
    await m.close()


# ── health tool ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_status():
    m = await _memory(_config())
    result = await _dispatch(m, "notmemory_health", {})
    assert result["status"] == "ok"
    assert result["initialized"] is True
    assert result["version"] == "0.1.0"
    assert "notmemory_retain" in result["tools"]
    await m.close()


# ── error handling ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_tool_raises():
    m = await _memory(_config())
    with pytest.raises(ValueError, match="Unknown tool"):
        await _dispatch(m, "unknown_tool", {})
    await m.close()
