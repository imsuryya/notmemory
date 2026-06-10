# Changelog

All notable changes to notmemory are documented here.

## [0.1.0] — 2025-06-01

### Added

**Core SDK**
- `AgentMemory` — main interface for all memory operations
- `retain()` — store memory entries with Pydantic schema validation
- `recall()` — keyword search via SQLite FTS5, temporal, hybrid strategies
- `rollback()` — tombstone any transaction to undo hallucinations
- `detect_conflicts()` — duplicate detection with health score 0-100
- `log_cycle_event()` — DAG audit trail per agent cycle
- `get_audit_trail()` — full forensic reconstruction
- `verify_integrity()` — SHA-256 hash chain verification
- `forget()` — GDPR-compliant tombstoning

**Storage**
- SQLite backend with FTS5 full-text search
- SHA-256 hash chaining on every write
- Async via aiosqlite + SQLAlchemy

**Adapters**
- `BaseAdapter` — abstract base class for all adapters
- `NotMemoryCheckpointer` — LangGraph drop-in for MemorySaver
- `NotMemoryChatHistory` — LangChain BaseChatMessageHistory
- `NotMemoryMem0Adapter` — Mem0 semantic search sidecar
- `NotMemorySuperMemoryAdapter` — SuperMemory semantic search sidecar
- MCP server with 5 tools for Claude Desktop / Cursor / Windsurf

**Memory**
- Confidence decay: c(t) = c0 * 2^(-t/30)
- `filter_fresh()` — filter stale entries by decay threshold
- `decay_score_entries()` — rank entries by current confidence

**Audit**
- `GitBackup` — periodic git commits of memory state
- `GitBackupConfig` — configurable interval, remote, branch

**Sync**
- `MemorySync` — sync memory banks between agent instances
- `PermissionStore` — READ/WRITE/ADMIN access control
- `AccessLevel` — ordered permission levels with wildcard support

**CI/CD**
- GitHub Actions matrix: Python 3.11, 3.12, 3.13
- Ruff linting and formatting
- Mypy type checking
- Dependabot weekly dependency updates
