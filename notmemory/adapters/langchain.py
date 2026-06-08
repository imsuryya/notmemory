# notmemory/adapters/langchain.py
"""
LangChain / LangGraph adapter for notmemory.

Provides:
  - ``NotMemoryCheckpointer``  — LangGraph checkpointer that persists graph
    state as auditable notmemory entries (drop-in for MemorySaver).
  - ``NotMemoryChatHistory``   — LangChain ``BaseChatMessageHistory`` that
    stores and retrieves chat messages with full audit trail.

Install:
    pip install -e ".[langchain]"
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from notmemory.adapters.base import BaseAdapter
from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig
from notmemory.memory.models import TrustLevel

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        RunnableConfig,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain extras are required. Install with: pip install -e '.[langchain]'"
    ) from exc


def _fts_safe(query: str) -> str:
    """Strip characters that break SQLite FTS5 MATCH expressions."""
    return re.sub(r"[\":*^()|\-]", " ", query).strip()


# ── checkpointer ──────────────────────────────────────────────────────────────


class NotMemoryCheckpointer(BaseAdapter, BaseCheckpointSaver):
    """
    LangGraph checkpointer backed by notmemory.

    Drop-in replacement for ``MemorySaver`` that adds:
    - SHA-256 hash-chained audit trail on every checkpoint write
    - ``rollback(transaction_id)`` to undo any checkpoint
    - Conflict detection across concurrent graph runs

    Usage::

        checkpointer = NotMemoryCheckpointer()
        graph = builder.compile(checkpointer=checkpointer)

        async with checkpointer:
            result = await graph.ainvoke(inputs, config)
    """

    BANK_ID = "langgraph-checkpoints"

    @property
    def adapter_name(self) -> str:
        return "langchain"

    @property
    def adapter_version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        memory: AgentMemory | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        BaseAdapter.__init__(self, memory=memory, config=config)

    async def initialize(self) -> None:
        await self._memory.initialize()
        self._initialized = True

    async def teardown(self) -> None:
        await self._memory.close()
        self._initialized = False

    async def __aenter__(self) -> NotMemoryCheckpointer:
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.teardown()

    # ── LangGraph checkpointer interface ──────────────────────────────

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        """Return the latest checkpoint for a thread, or None."""
        thread_id: str = config["configurable"].get("thread_id", "default")
        result = await self._memory.recall(
            bank_id=self.BANK_ID,
            query=_fts_safe(thread_id),
            strategies=["keyword"],
            limit=1,
        )
        if not result.entries:
            return None
        return result.entries[0].content.get("checkpoint")

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return a ``CheckpointTuple`` for LangGraph's internal protocol."""
        thread_id: str = config["configurable"].get("thread_id", "default")
        result = await self._memory.recall(
            bank_id=self.BANK_ID,
            query=_fts_safe(thread_id),
            strategies=["keyword"],
            limit=1,
        )
        if not result.entries:
            return None
        entry = result.entries[0]
        checkpoint: Checkpoint = entry.content["checkpoint"]
        metadata: CheckpointMetadata = entry.content.get("metadata", {})
        return CheckpointTuple(config=config, checkpoint=checkpoint, metadata=metadata)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        """Persist a checkpoint. Returns config with checkpoint_id filled in."""
        thread_id: str = config["configurable"].get("thread_id", "default")
        entry = await self._memory.retain(
            bank_id=self.BANK_ID,
            content={
                "thread_id": thread_id,
                "checkpoint": checkpoint,
                "metadata": metadata,
                "new_versions": new_versions,
            },
            context=thread_id,
            source="langgraph",
            trust_level="active",
        )
        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": entry.id,
            },
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Persist pending writes for a task (intermediate state)."""
        thread_id: str = config["configurable"].get("thread_id", "default")
        await self._memory.retain(
            bank_id=self.BANK_ID,
            content={
                "thread_id": thread_id,
                "task_id": task_id,
                "pending_writes": list(writes),
            },
            context=f"{thread_id}-pending",
            source="langgraph",
            trust_level="active",
        )

    async def alist(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        """Yield all checkpoints for a thread (newest first)."""
        thread_id: str = config["configurable"].get("thread_id", "default")
        result = await self._memory.recall(
            bank_id=self.BANK_ID,
            query=_fts_safe(thread_id),
            strategies=["keyword"],
            limit=limit or 20,
        )
        for entry in result.entries:
            checkpoint = entry.content.get("checkpoint")
            metadata = entry.content.get("metadata", {})
            if checkpoint:
                yield CheckpointTuple(config=config, checkpoint=checkpoint, metadata=metadata)

    def health(self) -> dict[str, Any]:
        return {**super().health(), "bank_id": self.BANK_ID}


# ── chat history ──────────────────────────────────────────────────────────────


class NotMemoryChatHistory(BaseAdapter, BaseChatMessageHistory):
    """
    LangChain ``BaseChatMessageHistory`` backed by notmemory.

    Usage::

        history = NotMemoryChatHistory(session_id="user-42")
        async with history:
            await history.aadd_messages(messages)
            msgs = await history.aget_messages()
    """

    @property
    def adapter_name(self) -> str:
        return "langchain"

    @property
    def adapter_version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        session_id: str,
        memory: AgentMemory | None = None,
        config: MemoryConfig | None = None,
        trust_level: TrustLevel = "active",
    ) -> None:
        BaseAdapter.__init__(self, memory=memory, config=config)
        self._session_id = session_id
        self._trust_level = trust_level

    @property
    def _bank_id(self) -> str:
        # hyphens only — no colons, no FTS special chars
        return f"langchain-chat-{self._session_id}"

    async def initialize(self) -> None:
        await self._memory.initialize()
        self._initialized = True

    async def teardown(self) -> None:
        await self._memory.close()
        self._initialized = False

    async def __aenter__(self) -> NotMemoryChatHistory:
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.teardown()

    # ── BaseChatMessageHistory interface ──────────────────────────────

    @property
    def messages(self) -> list[BaseMessage]:
        """Sync accessor — returns empty list; use ``aget_messages`` instead."""
        return []

    async def aget_messages(self) -> list[BaseMessage]:
        """Return all messages for this session, oldest first."""
        result = await self._memory.recall(
            bank_id=self._bank_id,
            strategies=["temporal"],
            limit=200,
        )
        messages: list[BaseMessage] = []
        for entry in result.entries:
            raw = entry.content.get("messages", [])
            messages.extend(messages_from_dict(raw))
        return messages

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist a batch of messages as a single auditable entry."""
        await self._memory.retain(
            bank_id=self._bank_id,
            content={"messages": messages_to_dict(list(messages))},
            context=self._session_id,
            source="langchain",
            trust_level=self._trust_level,
        )

    def clear(self) -> None:
        """Sync clear — no-op; use ``aclear()`` for async GDPR tombstoning."""

    async def aclear(self) -> None:
        """GDPR-tombstone all messages for this session."""
        await self._memory.forget(self._bank_id)

    def health(self) -> dict[str, Any]:
        return {
            **super().health(),
            "session_id": self._session_id,
            "bank_id": self._bank_id,
        }


__all__ = ["NotMemoryCheckpointer", "NotMemoryChatHistory"]
