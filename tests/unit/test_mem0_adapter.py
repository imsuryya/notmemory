# tests/unit/test_mem0_adapter.py
"""
Tests for the Mem0 sync-layer adapter.

Mem0 API calls are mocked — no real API key needed.
SQLite operations run against isolated in-memory DBs.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from notmemory.adapters.mem0 import NotMemoryMem0Adapter, _content_to_text
from notmemory.core.config import MemoryConfig


def _config() -> MemoryConfig:
    unique = uuid.uuid4().hex
    return MemoryConfig(
        db_url=f"sqlite+aiosqlite:///file:{unique}?mode=memory&cache=shared&uri=true"
    )


def _mock_mem0_client() -> MagicMock:
    """Return a mock AsyncMemoryClient."""
    client = MagicMock()
    client.add = AsyncMock(return_value={"id": "mem0-123"})
    client.search = AsyncMock(return_value=[])
    client.delete = AsyncMock(return_value=None)
    return client


# ── instantiation ─────────────────────────────────────────────────────────────


def test_adapter_metadata():
    adapter = NotMemoryMem0Adapter(mem0_api_key="test-key", config=_config())
    assert adapter.adapter_name == "mem0"
    assert adapter.adapter_version == "0.1.0"


def test_adapter_health_not_initialized():
    adapter = NotMemoryMem0Adapter(mem0_api_key="test-key", config=_config())
    h = adapter.health()
    assert h["adapter_name"] == "mem0"
    assert h["initialized"] is False
    assert h["mem0_connected"] is False
    assert h["api_key_set"] is True


def test_adapter_health_no_api_key():
    adapter = NotMemoryMem0Adapter(config=_config())
    h = adapter.health()
    assert h["api_key_set"] is False


# ── lifecycle ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_and_teardown():
    with patch("notmemory.adapters.mem0.AsyncMemoryClient") as mock_cls:
        mock_cls.return_value = _mock_mem0_client()
        adapter = NotMemoryMem0Adapter(mem0_api_key="test-key", config=_config())
        await adapter.initialize()
        assert adapter._initialized is True
        assert adapter._mem0 is not None
        await adapter.teardown()
        assert adapter._initialized is False
        assert adapter._mem0 is None


@pytest.mark.asyncio
async def test_context_manager():
    with patch("notmemory.adapters.mem0.AsyncMemoryClient") as mock_cls:
        mock_cls.return_value = _mock_mem0_client()
        async with NotMemoryMem0Adapter(mem0_api_key="test-key", config=_config()) as adapter:
            assert adapter._initialized is True
        assert adapter._initialized is False


# ── retain ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retain_writes_to_sqlite_and_mirrors_to_mem0():
    mock_client = _mock_mem0_client()
    with patch("notmemory.adapters.mem0.AsyncMemoryClient", return_value=mock_client):
        async with NotMemoryMem0Adapter(
            mem0_api_key="test-key", user_id="agent-1", config=_config()
        ) as adapter:
            entry = await adapter.retain(
                bank_id="facts",
                content={"fact": "sky is blue"},
            )
            assert entry.bank_id == "facts"
            assert entry.content == {"fact": "sky is blue"}
            # Mem0 mirror was called
            mock_client.add.assert_called_once()
            call_kwargs = mock_client.add.call_args.kwargs
            assert call_kwargs["user_id"] == "agent-1"
            assert call_kwargs["metadata"]["bank_id"] == "facts"
            assert call_kwargs["metadata"]["notmemory_id"] == entry.id


@pytest.mark.asyncio
async def test_retain_succeeds_even_if_mem0_fails():
    """SQLite write must succeed even when Mem0 mirror throws."""
    mock_client = _mock_mem0_client()
    mock_client.add = AsyncMock(side_effect=Exception("Mem0 down"))
    with patch("notmemory.adapters.mem0.AsyncMemoryClient", return_value=mock_client):
        async with NotMemoryMem0Adapter(mem0_api_key="test-key", config=_config()) as adapter:
            entry = await adapter.retain(
                bank_id="facts",
                content={"fact": "resilient write"},
            )
            # SQLite write succeeded despite Mem0 failure
            assert entry.id is not None


# ── recall ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_queries_sqlite():
    mock_client = _mock_mem0_client()
    with patch("notmemory.adapters.mem0.AsyncMemoryClient", return_value=mock_client):
        async with NotMemoryMem0Adapter(mem0_api_key="test-key", config=_config()) as adapter:
            result = await adapter.recall(bank_id="facts")
            assert result.entries == []
            assert result.bank_id == "facts"


@pytest.mark.asyncio
async def test_semantic_recall_queries_mem0():
    mock_client = _mock_mem0_client()
    mock_client.search = AsyncMock(
        return_value=[{"id": "m1", "memory": "sky is blue", "score": 0.95}]
    )
    with patch("notmemory.adapters.mem0.AsyncMemoryClient", return_value=mock_client):
        async with NotMemoryMem0Adapter(
            mem0_api_key="test-key", user_id="agent-1", config=_config()
        ) as adapter:
            results = await adapter.semantic_recall("color of sky")
            assert len(results) == 1
            assert results[0]["memory"] == "sky is blue"
            mock_client.search.assert_called_once_with(
                query="color of sky", user_id="agent-1", limit=10
            )


# ── helpers ───────────────────────────────────────────────────────────────────


def test_content_to_text_string_values():
    result = _content_to_text({"fact": "sky is blue", "source": "observation"})
    assert "sky is blue" in result
    assert "observation" in result


def test_content_to_text_mixed_values():
    result = _content_to_text({"count": 42, "label": "answer"})
    assert "answer" in result
    assert "42" in result


def test_content_to_text_empty():
    result = _content_to_text({})
    assert isinstance(result, str)
