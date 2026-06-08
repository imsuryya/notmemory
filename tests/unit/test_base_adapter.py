# tests/unit/test_base_adapter.py
"""Tests for the BaseAdapter abstract class."""

import pytest

from notmemory.adapters import BaseAdapter
from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig

# ── minimal concrete adapter for testing ──────────────────────────────────────


class _DummyAdapter(BaseAdapter):
    """Minimal concrete subclass that satisfies all abstract requirements."""

    @property
    def adapter_name(self) -> str:
        return "dummy"

    @property
    def adapter_version(self) -> str:
        return "0.1.0"

    async def initialize(self) -> None:
        self._initialized = True

    async def teardown(self) -> None:
        self._initialized = False


# ── tests ─────────────────────────────────────────────────────────────────────


def test_cannot_instantiate_base_adapter_directly():
    """BaseAdapter is abstract and must not be instantiatable."""
    with pytest.raises(TypeError):
        BaseAdapter()  # type: ignore[abstract]


def test_dummy_adapter_instantiation_with_default_config():
    adapter = _DummyAdapter()
    assert adapter.adapter_name == "dummy"
    assert adapter.adapter_version == "0.1.0"
    assert adapter._initialized is False


def test_dummy_adapter_instantiation_with_custom_config():
    config = MemoryConfig(agent_id="test-agent")
    adapter = _DummyAdapter(config=config)
    assert isinstance(adapter._memory, AgentMemory)


def test_dummy_adapter_instantiation_with_existing_memory():
    memory = AgentMemory(config=MemoryConfig(agent_id="pre-built"))
    adapter = _DummyAdapter(memory=memory)
    assert adapter._memory is memory


@pytest.mark.asyncio
async def test_initialize_sets_flag():
    adapter = _DummyAdapter()
    assert adapter._initialized is False
    await adapter.initialize()
    assert adapter._initialized is True


@pytest.mark.asyncio
async def test_teardown_clears_flag():
    adapter = _DummyAdapter()
    await adapter.initialize()
    await adapter.teardown()
    assert adapter._initialized is False


def test_health_returns_expected_keys():
    adapter = _DummyAdapter()
    h = adapter.health()
    assert h["adapter_name"] == "dummy"
    assert h["adapter_version"] == "0.1.0"
    assert h["initialized"] is False


def test_repr_not_initialized():
    adapter = _DummyAdapter()
    assert "not initialized" in repr(adapter)
    assert "dummy" in repr(adapter)


@pytest.mark.asyncio
async def test_repr_initialized():
    adapter = _DummyAdapter()
    await adapter.initialize()
    assert "initialized" in repr(adapter)
    assert "not initialized" not in repr(adapter)
