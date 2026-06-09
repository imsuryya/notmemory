# notmemory/audit/git_backup.py
"""
Git backup for notmemory.

Periodically commits the SQLite memory database to a local Git repository,
creating a time-series snapshot history of agent memory state.

Features:
  - Automatic periodic commits on a configurable interval
  - Manual snapshot on demand
  - Full commit history = complete memory timeline
  - Optional remote push (GitHub, GitLab, etc.)
  - Async-safe background task

Usage:
    from notmemory.audit.git_backup import GitBackup, GitBackupConfig

    backup = GitBackup(
        db_path="./notmemory.db",
        repo_path="./notmemory-backup",
        config=GitBackupConfig(
            commit_interval_seconds=300,
            remote="https://github.com/you/notmemory-backup.git",
        ),
    )

    async with backup:
        await your_agent_logic()

    # or manual snapshot
    await backup.snapshot(message="before risky operation")
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


class GitBackupConfig:
    """Configuration for GitBackup."""

    def __init__(
        self,
        commit_interval_seconds: int = 300,
        remote: str | None = None,
        branch: str = "main",
        author_name: str = "notmemory",
        author_email: str = "notmemory@localhost",
        push_on_commit: bool = False,
    ) -> None:
        """
        Args:
            commit_interval_seconds: How often to auto-commit (default 5 min).
            remote:                  Git remote URL for push (optional).
            branch:                  Branch name (default "main").
            author_name:             Git author name for commits.
            author_email:            Git author email for commits.
            push_on_commit:          Push to remote after every commit.
        """
        self.commit_interval_seconds = commit_interval_seconds
        self.remote = remote
        self.branch = branch
        self.author_name = author_name
        self.author_email = author_email
        self.push_on_commit = push_on_commit


class GitBackup:
    """
    Async git backup manager for notmemory SQLite databases.

    Copies the SQLite db into a dedicated git repo and commits
    on a configurable interval. Use as an async context manager
    for automatic background commits, or call ``snapshot()`` manually.
    """

    def __init__(
        self,
        db_path: str | Path,
        repo_path: str | Path,
        config: GitBackupConfig | None = None,
    ) -> None:
        """
        Args:
            db_path:   Path to the notmemory SQLite database file.
            repo_path: Path to the git backup repository (created if missing).
            config:    GitBackupConfig instance (optional).
        """
        self._db_path = Path(db_path)
        self._repo_path = Path(repo_path)
        self._config = config or GitBackupConfig()
        self._task: asyncio.Task | None = None
        self._running = False

    # ── lifecycle ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Set up the git repo if it doesn't exist."""
        self._repo_path.mkdir(parents=True, exist_ok=True)

        if not (self._repo_path / ".git").exists():
            self._git("init", "-b", self._config.branch)
            self._git("config", "user.name", self._config.author_name)
            self._git("config", "user.email", self._config.author_email)

            if self._config.remote:
                self._git("remote", "add", "origin", self._config.remote)

        self._running = True

    async def teardown(self) -> None:
        """Stop the background task cleanly."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def __aenter__(self) -> GitBackup:
        await self.initialize()
        self._task = asyncio.create_task(self._background_loop())
        return self

    async def __aexit__(self, *_) -> None:
        await self.snapshot(message="notmemory shutdown snapshot")
        await self.teardown()

    # ── public API ─────────────────────────────────────────────────────

    async def snapshot(self, message: str | None = None) -> str | None:
        """
        Copy the database and create a git commit immediately.

        Args:
            message: Commit message. Defaults to timestamped auto message.

        Returns:
            The git commit SHA, or None if nothing changed.
        """
        if not self._db_path.exists():
            return None

        dest = self._repo_path / self._db_path.name
        await asyncio.get_event_loop().run_in_executor(
            None, shutil.copy2, str(self._db_path), str(dest)
        )

        self._git("add", self._db_path.name)

        status = self._git_output("status", "--porcelain")
        if not status.strip():
            return None

        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        commit_msg = message or f"notmemory auto-backup {ts}"

        self._git("commit", "-m", commit_msg)

        if self._config.push_on_commit and self._config.remote:
            await self._push()

        return self._git_output("rev-parse", "HEAD").strip()

    async def history(self, limit: int = 20) -> list[dict]:
        """
        Return the last N commits as a list of dicts.

        Each dict has: sha, message, timestamp.
        """
        if not (self._repo_path / ".git").exists():
            return []

        try:
            raw = self._git_output(
                "log",
                f"-{limit}",
                "--pretty=format:%H|%s|%ai",
            )
        except subprocess.CalledProcessError:
            # Empty repo — no commits yet (exit code 128)
            return []

        if not raw.strip():
            return []

        commits = []
        for line in raw.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append(
                    {
                        "sha": parts[0],
                        "message": parts[1],
                        "timestamp": parts[2],
                    }
                )
        return commits

    async def restore(self, sha: str) -> bool:
        """
        Restore the database to the state at a specific commit SHA.

        Args:
            sha: Git commit SHA to restore from.

        Returns:
            True on success, False if SHA not found.
        """
        try:
            self._git("checkout", sha, "--", self._db_path.name)
            src = self._repo_path / self._db_path.name
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.copy2, str(src), str(self._db_path)
            )
            self._git("checkout", self._config.branch, "--", self._db_path.name)
            return True
        except subprocess.CalledProcessError:
            return False

    def health(self) -> dict:
        """Return backup health/state."""
        commits = 0
        latest_sha = None
        if (self._repo_path / ".git").exists():
            try:
                log = self._git_output("rev-list", "--count", "HEAD")
                commits = int(log.strip())
                latest_sha = self._git_output("rev-parse", "--short", "HEAD").strip()
            except (subprocess.CalledProcessError, ValueError):
                pass

        return {
            "running": self._running,
            "db_path": str(self._db_path),
            "repo_path": str(self._repo_path),
            "commit_interval_seconds": self._config.commit_interval_seconds,
            "remote": self._config.remote,
            "total_commits": commits,
            "latest_sha": latest_sha,
        }

    # ── internal ───────────────────────────────────────────────────────

    async def _background_loop(self) -> None:
        """Commit on the configured interval until stopped."""
        while self._running:
            await asyncio.sleep(self._config.commit_interval_seconds)
            if self._running:
                await self.snapshot()

    async def _push(self) -> None:
        """Push to remote (best-effort, never raises)."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self._repo_path),
                        "push",
                        "origin",
                        self._config.branch,
                    ],
                    check=True,
                    capture_output=True,
                ),
            )
        except Exception:  # noqa: BLE001
            pass

    def _git(self, *args: str) -> None:
        """Run a git command in the repo directory."""
        subprocess.run(
            ["git", "-C", str(self._repo_path), *args],
            check=True,
            capture_output=True,
        )

    def _git_output(self, *args: str) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            ["git", "-C", str(self._repo_path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


__all__ = ["GitBackup", "GitBackupConfig"]
