# tests/unit/test_supermemory_adapter.py
"""Tests for the SuperMemory sync-layer adapter. API calls fully mocked."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from notmemory.adapters.supermemory import (
    NotMemorySuperMemoryAdapter,
    _content_to_text,
)
from notmemory.core.config import MemoryConfig


def _config() -> MemoryConfig:
    unique = uuid.uuid4().hex
    return MemoryConfig(
        db_url=(f"sqlite+aiosqlite:///file:{unique}?mode=memory&cache=shared&uri=true")
    )


def _mock_response(
    status_code: int = 200,
    body: Any = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body or [])
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_response(200, {"id": "sm-123"}))
    client.get = AsyncMock(return_value=_mock_response(200, []))
    client.delete = AsyncMock(return_value=_mock_response(200))
    client.aclose = AsyncMock()
    return client


# ── instantiation ─────────────────────────────────────────────────────────────


def test_adapter_metadata():
    adapter = NotMemorySuperMemoryAdapter(api_key="sm_test", config=_config())
    assert adapter.adapter_name == "supermemory"
    assert adapter.adapter_version == "0.1.0"


def test_health_not_initialized():
    adapter = NotMemorySuperMemoryAdapter(api_key="sm_test", config=_config())
    h = adapter.health()
    assert h["adapter_name"] == "supermemory"
    assert h["initialized"] is False
    assert h["client_open"] is False
    assert h["api_key_set"] is True


def test_health_no_api_key():
    adapter = NotMemorySuperMemoryAdapter(config=_config())
    assert adapter.health()["api_key_set"] is False


# ── lifecycle ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_and_teardown():
    async with NotMemorySuperMemoryAdapter(api_key="sm_test", config=_config()) as adapter:
        assert adapter._initialized is True
        assert adapter._client is not None
    assert adapter._initialized is False
    assert adapter._client is None


@pytest.mark.asyncio
async def test_context_manager():
    async with NotMemorySuperMemoryAdapter(api_key="sm_test", config=_config()) as adapter:
        assert adapter._initialized is True
    assert adapter._initialized is False


# ── retain ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retain_writes_sqlite_and_mirrors_supermemory():
    async with NotMemorySuperMemoryAdapter(
        api_key="sm_test", user_id="agent-1", config=_config()
    ) as adapter:
        mock_client = _mock_client()
        adapter._client = mock_client

        entry = await adapter.retain(
            bank_id="facts",
            content={"fact": "sky is blue"},
        )

        assert entry.bank_id == "facts"
        assert entry.content == {"fact": "sky is blue"}
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
        assert body["userId"] == "agent-1"
        assert body["metadata"]["bank_id"] == "facts"
        assert body["metadata"]["notmemory_id"] == entry.id


@pytest.mark.asyncio
async def test_retain_succeeds_if_supermemory_fails():
    """SQLite write must succeed even when SuperMemory mirror throws."""
    async with NotMemorySuperMemoryAdapter(api_key="sm_test", config=_config()) as adapter:
        mock_client = _mock_client()
        mock_client.post = AsyncMock(side_effect=Exception("SM down"))
        adapter._client = mock_client

        entry = await adapter.retain(
            bank_id="facts",
            content={"fact": "resilient"},
        )
        assert entry.id is not None


# ── recall ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_queries_sqlite():
    async with NotMemorySuperMemoryAdapter(api_key="sm_test", config=_config()) as adapter:
        result = await adapter.recall(bank_id="facts")
        assert result.entries == []
        assert result.bank_id == "facts"


@pytest.mark.asyncio
async def test_semantic_recall_queries_supermemory():
    async with NotMemorySuperMemoryAdapter(
        api_key="sm_test", user_id="agent-1", config=_config()
    ) as adapter:
        mock_client = _mock_client()
        mock_client.get = AsyncMock(
            return_value=_mock_response(
                200, [{"id": "sm-1", "content": "sky is blue", "score": 0.95}]
            )
        )
        adapter._client = mock_client

        results = await adapter.semantic_recall("color of sky")
        assert len(results) == 1
        assert results[0]["content"] == "sky is blue"
        mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_semantic_recall_returns_empty_on_error():
    async with NotMemorySuperMemoryAdapter(api_key="sm_test", config=_config()) as adapter:
        mock_client = _mock_client()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        adapter._client = mock_client

        results = await adapter.semantic_recall("anything")
        assert results == []


# ── helpers ───────────────────────────────────────────────────────────────────


def test_content_to_text_strings():
    result = _content_to_text({"fact": "sky is blue", "src": "obs"})
    assert "sky is blue" in result
    assert "obs" in result


def test_content_to_text_mixed():
    result = _content_to_text({"count": 42, "label": "answer"})
    assert "answer" in result


def test_content_to_text_empty():
    result = _content_to_text({})
    assert isinstance(result, str)
