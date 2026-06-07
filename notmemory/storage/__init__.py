from notmemory.core.config import MemoryConfig
from notmemory.storage.backends.base import BaseStorageBackend
from notmemory.storage.backends.sqlite import SQLiteBackend


def create_backend(config: MemoryConfig) -> BaseStorageBackend:
    if config.storage == "sqlite":
        return SQLiteBackend(db_url=config.db_url, hash_chaining=config.hash_chain.enabled)
    raise ValueError(f"Unknown storage backend: {config.storage!r}")


__all__ = ["BaseStorageBackend", "SQLiteBackend", "create_backend"]