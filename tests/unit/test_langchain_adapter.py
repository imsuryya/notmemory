# tests/unit/test_langchain_adapter.py
"""Tests for the LangChain adapter (checkpointer + chat history)."""

import uuid

import pytest

pytest.importorskip("langchain_core", reason="langchain extras not installed")
pytest.importorskip("langgraph", reason="langchain extras not installed")

from notmemory.adapters.langchain import NotMemoryChatHistory, NotMemoryCheckpointer
from notmemory.core.config import MemoryConfig


def _config() -> MemoryConfig:
    """Each call gets a unique named in-memory DB — no pool sharing."""
    unique = uuid.uuid4().hex
    return MemoryConfig(
        db_url=f"sqlite+aiosqlite:///file:{unique}?mode=memory&cache=shared&uri=true"
    )


# ── NotMemoryCheckpointer ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpointer_initialize_teardown():
    async with NotMemoryCheckpointer(config=_config()) as cp:
        assert cp._initialized is True
    assert cp._initialized is False


@pytest.mark.asyncio
async def test_checkpointer_metadata():
    cp = NotMemoryCheckpointer(config=_config())
    assert cp.adapter_name == "langchain"
    assert cp.adapter_version == "0.1.0"
    assert cp.BANK_ID == "langgraph-checkpoints"


@pytest.mark.asyncio
async def test_checkpointer_aget_returns_none_when_empty():
    async with NotMemoryCheckpointer(config=_config()) as cp:
        result = await cp.aget({"configurable": {"thread_id": "thread-1"}})
        assert result is None


@pytest.mark.asyncio
async def test_checkpointer_aget_tuple_returns_none_when_empty():
    async with NotMemoryCheckpointer(config=_config()) as cp:
        result = await cp.aget_tuple({"configurable": {"thread_id": "thread-1"}})
        assert result is None


@pytest.mark.asyncio
async def test_checkpointer_aput_returns_config_with_checkpoint_id():
    async with NotMemoryCheckpointer(config=_config()) as cp:
        cfg = {"configurable": {"thread_id": "thread-1"}}
        checkpoint = {
            "v": 1,
            "ts": "2024-01-01",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        }
        result_cfg = await cp.aput(cfg, checkpoint, {"step": 1}, {})
        assert "checkpoint_id" in result_cfg["configurable"]
        assert result_cfg["configurable"]["thread_id"] == "thread-1"


@pytest.mark.asyncio
async def test_checkpointer_health():
    cp = NotMemoryCheckpointer(config=_config())
    h = cp.health()
    assert h["adapter_name"] == "langchain"
    assert h["bank_id"] == "langgraph-checkpoints"
    assert h["initialized"] is False


# ── NotMemoryChatHistory ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_history_initialize_teardown():
    async with NotMemoryChatHistory("sess-1", config=_config()) as h:
        assert h._initialized is True
    assert h._initialized is False


@pytest.mark.asyncio
async def test_chat_history_empty_on_start():
    async with NotMemoryChatHistory("sess-1", config=_config()) as h:
        msgs = await h.aget_messages()
        assert msgs == []


@pytest.mark.asyncio
async def test_chat_history_add_and_retrieve():
    from langchain_core.messages import AIMessage, HumanMessage

    async with NotMemoryChatHistory("sess-2", config=_config()) as h:
        await h.aadd_messages([HumanMessage(content="hello"), AIMessage(content="hi")])
        msgs = await h.aget_messages()
        assert len(msgs) == 2
        assert msgs[0].content == "hello"
        assert msgs[1].content == "hi"


@pytest.mark.asyncio
async def test_chat_history_aclear():
    from langchain_core.messages import HumanMessage

    async with NotMemoryChatHistory("sess-3", config=_config()) as h:
        await h.aadd_messages([HumanMessage(content="delete me")])
        await h.aclear()
        msgs = await h.aget_messages()
        assert msgs == []


@pytest.mark.asyncio
async def test_chat_history_health():
    h = NotMemoryChatHistory("sess-99", config=_config())
    health = h.health()
    assert health["session_id"] == "sess-99"
    assert health["bank_id"] == "langchain-chat-sess-99"
