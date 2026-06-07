from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class HashChainConfig(BaseModel):
    enabled: bool = True
    algorithm: Literal["sha256", "sha512"] = "sha256"


class GitBackupConfig(BaseModel):
    enabled: bool = False
    remote: str | None = None
    commit_interval_seconds: int = 300


class RetentionConfig(BaseModel):
    confidence_half_life_days: float = 30.0
    deprecation_threshold: float = 0.05


class MemoryConfig(BaseModel):
    storage: Literal["sqlite", "postgres"] = "sqlite"
    db_url: str = "sqlite+aiosqlite:///./notmemory.db"
    hash_chain: HashChainConfig = Field(default_factory=HashChainConfig)
    git_backup: GitBackupConfig = Field(default_factory=GitBackupConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    @model_validator(mode="after")
    def _validate(self) -> "MemoryConfig":
        if self.storage == "postgres" and "sqlite" in self.db_url:
            raise ValueError("storage=postgres but db_url looks like SQLite.")
        return self