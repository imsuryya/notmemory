# notmemory

> Deterministic memory observability for long-lived AI agents.

[![CI](https://github.com/notmemory/notmemory/actions/workflows/ci.yml/badge.svg)](https://github.com/notmemory/notmemory/actions)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/notmemory/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-0.1.0-orange)](https://pypi.org/project/notmemory/)

---

## What is notmemory?

notmemory is an open-source Python SDK that gives AI agents
**auditable, reversible, GDPR-compliant memory**.

Every memory write is cryptographically hash-chained — like Git commits
for agent state. Every mistake can be rolled back. Every deletion leaves
a forensic trail. Every read is logged.

Built for production AI systems where trust, compliance, and
debuggability are non-negotiable.

---

## Why notmemory?

| Problem | notmemory solution |
|---|---|
| Agent hallucinates and corrupts memory | `rollback(transaction_id)` undoes it |
| GDPR deletion request | `forget(bank_id)` tombstones with audit trail |
| Can't debug what agent remembered | `get_audit_trail(cycle_id)` forensic replay |
| Memory grows stale over time | Confidence decay `c = c₀ · 2^(-t/30)` |
| Need semantic search too | Mem0 / SuperMemory sync-layer adapters |
| Want Claude/Cursor to use agent memory | MCP server adapter |

---

## Features

- **Cryptographic audit trail** — SHA-256 hash chaining on every write
- **Git-like rollback** — undo any transaction by ID
- **Conflict detection** — duplicate detection with health score 0–100
- **GDPR tombstoning** — compliant deletion with forensic trail
- **Full-text search** — SQLite FTS5 keyword search
- **Confidence decay** — exponential half-life `c = c₀ · 2^(-t/30)`
- **LangGraph checkpointer** — drop-in replacement for `MemorySaver`
- **LangChain chat history** — `BaseChatMessageHistory` implementation
- **Mem0 sidecar** — semantic search via Mem0
- **SuperMemory sidecar** — semantic search via SuperMemory
- **MCP server** — Claude Desktop / Cursor / Windsurf integration

---

## Install

```bash
# Core SDK
pip install notmemory

# With LangChain / LangGraph
pip install "notmemory[langchain]"

# With Mem0 semantic search
pip install "notmemory[mem0]" mem0ai

# With MCP server
pip install "notmemory[mcp]"

# Everything
pip install "notmemory[langchain,mcp]" mem0ai
```

---

## Quickstart

```python
import asyncio
from notmemory import AgentMemory, MemoryConfig

async def main():
    async with AgentMemory() as memory:

        # Store a memory
        entry = await memory.retain(
            bank_id="facts",
            content={"fact": "The Eiffel Tower is in Paris"},
            source="user",
        )
        print(f"Stored: {entry.id}")

        # Search memory
        result = await memory.recall(
            bank_id="facts",
            query="Eiffel Tower",
        )
        for e in result.entries:
            print(e.content)

        # Rollback a bad write
        await memory.rollback(entry.transaction_id)

        # GDPR delete entire bank
        await memory.forget("facts")

asyncio.run(main())
```

---

## Core API

### AgentMemory

```python
from notmemory import AgentMemory, MemoryConfig

config = MemoryConfig(db_url="sqlite+aiosqlite:///./myagent.db")

async with AgentMemory(config=config) as memory:
    ...
```

| Method | Returns | Description |
|--------|---------|-------------|
| `retain(*, bank_id, content, ...)` | `MemoryEntry` | Store with hash chain |
| `recall(*, bank_id, query, ...)` | `RecallResult` | Keyword / temporal search |
| `rollback(transaction_id)` | `RollbackResult` | Tombstone a transaction |
| `detect_conflicts(*, bank_id)` | `ConflictReport` | Find duplicates, health score |
| `log_cycle_event(*, cycle_id, ...)` | `CycleEvent` | DAG audit event |
| `get_audit_trail(*, cycle_id)` | `AuditTrail` | Forensic reconstruction |
| `verify_integrity(bank_id)` | `bool` | Verify SHA-256 hash chain |
| `forget(bank_id, *, entry_ids)` | `int` | GDPR tombstone |

---

## Confidence Decay

Memories lose confidence over time using exponential half-life decay:
c(t) = c₀ · 2^(-t / half_life_days)

| Age | Confidence (from 1.0) |
|-----|-----------------------|
| 0 days | 1.000 |
| 30 days | 0.500 |
| 60 days | 0.250 |
| 90 days | 0.125 |

```python
from notmemory.memory.decay import decayed_confidence, filter_fresh

# Current decayed score
score = decayed_confidence(entry, half_life_days=30.0)

# Filter stale entries out, sort fresh ones by score
fresh = filter_fresh(entries, half_life_days=30.0, deprecation_threshold=0.05)
```

---

## Adapters

### LangChain / LangGraph

```python
from notmemory.adapters.langchain import (
    NotMemoryCheckpointer,
    NotMemoryChatHistory,
)

# Drop-in for MemorySaver — adds hash-chained audit trail
checkpointer = NotMemoryCheckpointer()
graph = builder.compile(checkpointer=checkpointer)

async with checkpointer:
    result = await graph.ainvoke(inputs, config)

# Chat history with GDPR tombstoning
history = NotMemoryChatHistory(session_id="user-42")
async with history:
    await history.aadd_messages(messages)
    msgs = await history.aget_messages()
```

### Mem0

```python
from notmemory.adapters.mem0 import NotMemoryMem0Adapter
import os

adapter = NotMemoryMem0Adapter(
    mem0_api_key=os.environ["MEM0_API_KEY"],
    user_id="agent-1",
)
async with adapter:
    # Writes to SQLite + mirrors to Mem0
    await adapter.retain(bank_id="facts", content={"fact": "..."})

    # Keyword search (SQLite)
    result = await adapter.recall(bank_id="facts", query="Paris")

    # Semantic search (Mem0)
    results = await adapter.semantic_recall("European capitals")
```

### SuperMemory

```python
from notmemory.adapters.supermemory import NotMemorySuperMemoryAdapter
import os

adapter = NotMemorySuperMemoryAdapter(
    api_key=os.environ["SUPERMEMORY_API_KEY"],
    user_id="agent-1",
)
async with adapter:
    await adapter.retain(bank_id="facts", content={"fact": "..."})
    results = await adapter.semantic_recall("European capitals")
```

### MCP Server (Claude Desktop / Cursor / Windsurf)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "notmemory": {
      "command": "python",
      "args": ["-m", "notmemory.adapters.mcp"],
      "env": {
        "NOTMEMORY_DB_URL": "sqlite+aiosqlite:///./notmemory.db"
      }
    }
  }
}
```

Available tools: `notmemory_retain`, `notmemory_recall`,
`notmemory_rollback`, `notmemory_forget`, `notmemory_health`

---

## Configuration

```python
from notmemory import MemoryConfig
from notmemory.core.config import HashChainConfig, RetentionConfig

config = MemoryConfig(
    storage="sqlite",
    db_url="sqlite+aiosqlite:///./agent.db",
    hash_chain=HashChainConfig(
        enabled=True,
        algorithm="sha256",
    ),
    retention=RetentionConfig(
        confidence_half_life_days=30.0,
        deprecation_threshold=0.05,
    ),
)
```

---

## Project Structure
notmemory/
├── core/
│   ├── agent_memory.py     # AgentMemory — main interface
│   ├── config.py           # MemoryConfig (Pydantic v2)
│   └── exceptions.py       # exception hierarchy
├── memory/
│   ├── models.py           # domain models
│   └── decay.py            # confidence decay
├── storage/
│   └── backends/
│       ├── base.py         # abstract storage interface
│       └── sqlite.py       # SQLite + FTS5 + hash chaining
└── adapters/
├── base.py             # BaseAdapter abstract class
├── langchain.py        # LangChain / LangGraph
├── mem0.py             # Mem0 sync-layer
├── supermemory.py      # SuperMemory sync-layer
└── mcp.py              # MCP server

---

## Development

```bash
git clone https://github.com/notmemory/notmemory
cd notmemory

python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev,langchain,mcp]" mem0ai

# Before every commit
ruff check . --fix
ruff format .
pytest --tb=short
```

---

## Contributing

1. Fork the repo
2. Branch from `develop`: `git checkout -b feature/your-feature`
3. Make changes, run `ruff` + `pytest`
4. Open PR targeting `develop`
5. CI must be green on Python 3.11, 3.12, 3.13

---

## License

MIT — see [LICENSE](LICENSE).