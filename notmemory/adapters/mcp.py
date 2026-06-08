# notmemory/adapters/mcp.py
"""
MCP server adapter for notmemory.

Exposes notmemory's core operations as MCP tools so Claude, Cursor,
Windsurf, and any MCP-compatible client can read/write agent memory
with full audit trails.

Tools exposed:
  - notmemory_retain   — store a memory entry
  - notmemory_recall   — keyword/temporal search
  - notmemory_rollback — undo a transaction (tombstone)
  - notmemory_forget   — GDPR-compliant bulk tombstone
  - notmemory_health   — server health check

Install:
    pip install -e ".[mcp]"

Run as stdio server (for Claude Desktop / Cursor):
    python -m notmemory.adapters.mcp

Run as HTTP/SSE server:
    python -m notmemory.adapters.mcp --transport sse --port 8765
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from notmemory.core.agent_memory import AgentMemory
from notmemory.core.config import MemoryConfig

# ── tool schemas ──────────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="notmemory_retain",
        description=(
            "Store a memory entry with full cryptographic audit trail. "
            "Returns the entry id and transaction_id."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bank_id": {
                    "type": "string",
                    "description": "Memory bank namespace, e.g. 'facts' or 'user-prefs'",
                },
                "content": {
                    "type": "object",
                    "description": "Arbitrary JSON content to store",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context label",
                },
                "source": {
                    "type": "string",
                    "description": "Optional source label, e.g. 'user', 'agent'",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0–1.0 (default 1.0)",
                    "default": 1.0,
                },
            },
            "required": ["bank_id", "content"],
        },
    ),
    Tool(
        name="notmemory_recall",
        description=(
            "Search memory entries by keyword or retrieve recent entries. "
            "Returns a list of matching entries with content and metadata."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bank_id": {
                    "type": "string",
                    "description": "Memory bank to search",
                },
                "query": {
                    "type": "string",
                    "description": "Keyword search query (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["bank_id"],
        },
    ),
    Tool(
        name="notmemory_rollback",
        description=(
            "Undo a memory transaction by tombstoning it. "
            "Use this to correct hallucinations or bad writes. "
            "Requires the transaction_id from a previous retain call."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Transaction ID to roll back",
                },
            },
            "required": ["transaction_id"],
        },
    ),
    Tool(
        name="notmemory_forget",
        description=(
            "GDPR-compliant tombstone of memory entries. "
            "Provide entry_ids to delete specific entries, "
            "or omit to tombstone the entire bank."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bank_id": {
                    "type": "string",
                    "description": "Memory bank to tombstone",
                },
                "entry_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific entry IDs to delete (optional)",
                },
            },
            "required": ["bank_id"],
        },
    ),
    Tool(
        name="notmemory_health",
        description="Return notmemory server health and configuration.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ── server ────────────────────────────────────────────────────────────────────


def create_server(config: MemoryConfig | None = None) -> Server:
    """
    Create and return a configured MCP Server instance.

    Args:
        config: MemoryConfig to use. Defaults to SQLite at ./notmemory.db
    """
    app = Server("notmemory")
    memory = AgentMemory(config=config or MemoryConfig())

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        # Initialize on first tool call
        if not memory._initialized:
            await memory.initialize()

        try:
            result = await _dispatch(memory, name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as exc:  # noqa: BLE001
            error = {"error": str(exc), "tool": name}
            return [TextContent(type="text", text=json.dumps(error, indent=2))]

    return app


async def _dispatch(
    memory: AgentMemory,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Route a tool call to the correct AgentMemory method."""

    if name == "notmemory_retain":
        entry = await memory.retain(
            bank_id=args["bank_id"],
            content=args["content"],
            context=args.get("context"),
            source=args.get("source"),
            confidence=float(args.get("confidence", 1.0)),
        )
        return {
            "id": entry.id,
            "bank_id": entry.bank_id,
            "transaction_id": entry.transaction_id,
            "timestamp": entry.timestamp.isoformat(),
            "confidence": entry.confidence,
            "trust_level": entry.trust_level,
        }

    if name == "notmemory_recall":
        result = await memory.recall(
            bank_id=args["bank_id"],
            query=args.get("query"),
            limit=int(args.get("limit", 10)),
        )
        return {
            "bank_id": result.bank_id,
            "query": result.query,
            "token_count": result.token_count,
            "elapsed_ms": result.elapsed_ms,
            "entries": [
                {
                    "id": e.id,
                    "content": e.content,
                    "context": e.context,
                    "source": e.source,
                    "confidence": e.confidence,
                    "trust_level": e.trust_level,
                    "timestamp": e.timestamp.isoformat(),
                    "transaction_id": e.transaction_id,
                }
                for e in result.entries
            ],
        }

    if name == "notmemory_rollback":
        result = await memory.rollback(args["transaction_id"])
        return {
            "transaction_id": result.transaction_id,
            "entries_reversed": result.entries_reversed,
            "success": result.success,
        }

    if name == "notmemory_forget":
        count = await memory.forget(
            args["bank_id"],
            entry_ids=args.get("entry_ids"),
        )
        return {"bank_id": args["bank_id"], "entries_tombstoned": count}

    if name == "notmemory_health":
        return {
            "status": "ok",
            "initialized": memory._initialized,
            "storage": memory._config.storage,
            "tools": [t.name for t in TOOLS],
            "version": "0.1.0",
        }

    raise ValueError(f"Unknown tool: {name!r}")


# ── entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Run notmemory as a stdio MCP server."""
    config = MemoryConfig()
    server = create_server(config)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
