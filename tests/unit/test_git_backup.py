# tests/unit/test_git_backup.py
"""Tests for GitBackup — all tests use temp directories, no real DB needed."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from notmemory.audit.git_backup import GitBackup, GitBackupConfig


def _force_rmtree(path: Path) -> None:
    """Remove directory tree — handles Windows read-only git object files."""

    def _on_error(func, fpath, _exc):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_error)
    else:
        shutil.rmtree(path, onerror=_on_error)


def _temp_dir() -> Path:
    return Path(tempfile.mkdtemp())


def _make_db(path: Path) -> Path:
    """Create a fake SQLite db file."""
    db = path / "notmemory.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    return db


# ── config ────────────────────────────────────────────────────────────────────


def test_default_config():
    config = GitBackupConfig()
    assert config.commit_interval_seconds == 300
    assert config.remote is None
    assert config.branch == "main"
    assert config.push_on_commit is False


def test_custom_config():
    config = GitBackupConfig(
        commit_interval_seconds=60,
        remote="https://github.com/test/repo.git",
        branch="backup",
        push_on_commit=True,
    )
    assert config.commit_interval_seconds == 60
    assert config.remote == "https://github.com/test/repo.git"
    assert config.push_on_commit is True


# ── initialize ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_creates_git_repo():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        assert (repo / ".git").exists()
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


@pytest.mark.asyncio
async def test_initialize_idempotent():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        await backup.initialize()
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


# ── snapshot ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_creates_commit():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        sha = await backup.snapshot(message="test snapshot")
        assert sha is not None
        assert len(sha) == 40
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


@pytest.mark.asyncio
async def test_snapshot_returns_none_when_no_changes():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        await backup.snapshot(message="first")
        result = await backup.snapshot(message="second")
        assert result is None
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


@pytest.mark.asyncio
async def test_snapshot_returns_none_if_db_missing():
    tmp = _temp_dir()
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=tmp / "missing.db", repo_path=repo)
        await backup.initialize()
        result = await backup.snapshot()
        assert result is None
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


# ── history ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_after_snapshots():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        await backup.snapshot(message="snap 1")

        db.write_bytes(b"SQLite format 3\x00" + b"\x01" * 100)
        await backup.snapshot(message="snap 2")

        history = await backup.history(limit=10)
        assert len(history) == 2
        assert history[0]["message"] == "snap 2"
        assert "sha" in history[0]
        assert "timestamp" in history[0]
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


@pytest.mark.asyncio
async def test_history_empty_repo():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        history = await backup.history()
        assert history == []
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


# ── health ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_before_snapshot():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        h = backup.health()
        assert h["running"] is True
        assert h["total_commits"] == 0
        assert h["remote"] is None
        await backup.teardown()
    finally:
        _force_rmtree(tmp)


@pytest.mark.asyncio
async def test_health_after_snapshot():
    tmp = _temp_dir()
    db = _make_db(tmp)
    repo = tmp / "backup"
    try:
        backup = GitBackup(db_path=db, repo_path=repo)
        await backup.initialize()
        await backup.snapshot(message="first")
        h = backup.health()
        assert h["total_commits"] == 1
        assert h["latest_sha"] is not None
        await backup.teardown()
    finally:
        _force_rmtree(tmp)
