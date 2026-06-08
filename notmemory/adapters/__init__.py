# notmemory/adapters/__init__.py
"""
Adapter layer for notmemory.

All adapters inherit from BaseAdapter, which provides:
  - lifecycle hooks (initialize / teardown)
  - a wrapped AgentMemory instance accessible as ``self._memory``
  - health-check and metadata introspection
"""

from __future__ import annotations

import abc
from typing import Any

from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig


class BaseAdapter(abc.ABC):
    """
    Abstract base class every notmemory adapter must inherit from.

    Subclasses implement:
      - ``adapter_name``    — unique lowercase slug, e.g. ``"langchain"``
      - ``adapter_version`` — semver string, e.g. ``"0.1.0"``
      - ``initialize()``    — async setup (connect to external service, etc.)
      - ``teardown()``      — async cleanup (close connections, flush buffers)

    The wrapped ``AgentMemory`` instance is available as ``self._memory``.
    Adapters stay thin; the core SDK remains the single source of truth.
    """

    # ── subclass-defined metadata ──────────────────────────────────────

    @property
    @abc.abstractmethod
    def adapter_name(self) -> str:
        """Unique lowercase slug, e.g. ``'langchain'``."""

    @property
    @abc.abstractmethod
    def adapter_version(self) -> str:
        """Semver string, e.g. ``'0.1.0'``."""

    # ── lifecycle ──────────────────────────────────────────────────────

    @abc.abstractmethod
    async def initialize(self) -> None:
        """
        Async setup hook called once before the adapter is used.

        Use for: connecting to external services, loading credentials,
        registering hooks with the host framework (LangGraph, etc.).
        """

    @abc.abstractmethod
    async def teardown(self) -> None:
        """
        Async cleanup hook called when the adapter is no longer needed.

        Use for: closing HTTP sessions, flushing write buffers,
        deregistering callbacks.
        """

    # ── construction ───────────────────────────────────────────────────

    def __init__(
        self,
        memory: AgentMemory | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """
        Args:
            memory: An already-constructed ``AgentMemory`` instance.
                    If *None*, a new one is created from ``config``.
            config: ``MemoryConfig`` passed to a freshly created
                    ``AgentMemory``. Ignored when ``memory`` is supplied.
        """
        if memory is not None:
            self._memory = memory
        else:
            self._memory = AgentMemory(config=config or MemoryConfig())
        self._initialized = False

    # ── introspection ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """
        Return a dict describing the adapter's current health/state.

        Override in subclasses to add framework-specific diagnostics.
        The base dict always includes ``adapter_name``, ``adapter_version``,
        and ``initialized``.
        """
        return {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"<{self.__class__.__name__} adapter_name={self.adapter_name!r} {status}>"


__all__ = ["BaseAdapter"]
