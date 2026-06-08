# notmemory/adapters/base.py
"""BaseAdapter — abstract base class all notmemory adapters inherit from."""

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

    @property
    @abc.abstractmethod
    def adapter_name(self) -> str:
        """Unique lowercase slug, e.g. ``'langchain'``."""

    @property
    @abc.abstractmethod
    def adapter_version(self) -> str:
        """Semver string, e.g. ``'0.1.0'``."""

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Async setup hook — connect, load credentials, register hooks."""

    @abc.abstractmethod
    async def teardown(self) -> None:
        """Async cleanup hook — close sessions, flush buffers."""

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

    def health(self) -> dict[str, Any]:
        """Return adapter health/state dict. Override to extend."""
        return {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"<{self.__class__.__name__} adapter_name={self.adapter_name!r} {status}>"


__all__ = ["BaseAdapter"]
