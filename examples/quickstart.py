import asyncio

from notmemory import AgentMemory, MemoryConfig


async def main() -> None:
    async with AgentMemory(MemoryConfig(db_url="sqlite+aiosqlite:///./quickstart.db")) as memory:
        print("=== 1. RETAIN ===")
        entry = await memory.retain(
            bank_id="user",
            content={"name": "Alice", "role": "engineer"},
            context="onboarding",
        )
        print(f"stored: {entry.id}  txn={entry.transaction_id}")

        print("\n=== 2. RECALL ===")
        results = await memory.recall(bank_id="user", query="Alice")
        for e in results.entries:
            print(f"  {e.content}")

        print("\n=== 3. AUDIT TRAIL ===")
        await memory.log_cycle_event(
            cycle_id="cyc-001",
            event_type="llm-call",
            model="claude-3-5-sonnet",
            tokens_in=100,
            tokens_out=200,
            elapsed_ms=500.0,
        )
        trail = await memory.get_audit_trail(cycle_id="cyc-001")
        print(f"  events: {len(trail.events)}  tokens: {trail.total_tokens}")

        print("\n=== 4. HASH CHAIN ===")
        ok = await memory.verify_integrity("user")
        print(f"  intact: {ok}")

        print("\n=== 5. ROLLBACK ===")
        bad = await memory.retain(
            bank_id="user",
            content={"name": "HALLUCINATED"},
        )
        result = await memory.rollback(bad.transaction_id)
        print(f"  reversed: {result.entries_reversed}")

        print("\n=== 6. CONFLICTS ===")
        await memory.retain(bank_id="facts", content={"x": 1})
        await memory.retain(bank_id="facts", content={"x": 1})
        report = await memory.detect_conflicts(bank_id="facts")
        print(f"  health: {report.health_score}/100  conflicts: {len(report.conflicts)}")

        print("\n=== 7. FORGET ===")
        count = await memory.forget("facts")
        print(f"  tombstoned: {count}")


if __name__ == "__main__":
    asyncio.run(main())
