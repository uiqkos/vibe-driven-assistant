"""Persistent local repo management: clone, fetch, branch, commit, push."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKDIR = Path.home() / ".coding-agent"


class RepoManager:
    def __init__(
        self,
        repo_name: str,
        github_token: str,
        workdir: Path | None = None,
    ) -> None:
        self.repo_name = repo_name  # "owner/repo"
        base = workdir or Path(os.environ.get("CODING_AGENT_WORKDIR", str(DEFAULT_WORKDIR)))
        self.repo_dir = base / "repos" / repo_name / "source"
        self.index_dir = base / "repos" / repo_name / ".code_index"
        self.rag_index_dir = base / "repos" / repo_name / ".rag_index"
        self._token = github_token

    @property
    def _clone_url(self) -> str:
        return f"https://x-access-token:{self._token}@github.com/{self.repo_name}.git"

    def _run(self, cmd: list[str], retries: int = 0, **kwargs) -> subprocess.CompletedProcess:
        logger.debug("Running: %s", " ".join(cmd))
        for attempt in range(retries + 1):
            try:
                return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
            except subprocess.CalledProcessError as exc:
                # Sanitize token from logged output
                safe_cmd = " ".join(cmd).replace(self._token, "***") if self._token else " ".join(cmd)
                stderr = (exc.stderr or "").replace(self._token, "***") if self._token else (exc.stderr or "")
                if attempt < retries:
                    delay = 2 ** attempt
                    logger.warning("Command failed (attempt %d/%d), retrying in %ds: %s\nstderr: %s",
                                   attempt + 1, retries + 1, delay, safe_cmd, stderr)
                    time.sleep(delay)
                else:
                    logger.error("Command failed after %d attempts: %s\nstderr: %s",
                                 retries + 1, safe_cmd, stderr)
                    raise
        raise AssertionError("unreachable")  # retries loop always returns or raises

    def _configure_git(self) -> None:
        """Set git user identity if not already configured."""
        try:
            self._run(["git", "config", "user.name"], cwd=self.repo_dir)
        except subprocess.CalledProcessError:
            self._run(["git", "config", "user.name", "coding-agent"], cwd=self.repo_dir)
            self._run(["git", "config", "user.email", "coding-agent@noreply"], cwd=self.repo_dir)

    def _cleanup_lock_files(self) -> None:
        """Remove stale git lock files that block operations."""
        for lock in self.repo_dir.rglob(".git/*.lock"):
            logger.warning("Removing stale lock file: %s", lock)
            lock.unlink(missing_ok=True)
        # Also check nested lock files (e.g. .git/refs/heads/*.lock)
        for lock in self.repo_dir.rglob(".git/**/*.lock"):
            logger.warning("Removing stale lock file: %s", lock)
            lock.unlink(missing_ok=True)

    def _reclone(self) -> None:
        """Delete the repo directory and clone from scratch."""
        import shutil

        logger.warning("Re-cloning %s (deleting %s)", self.repo_name, self.repo_dir)
        shutil.rmtree(self.repo_dir, ignore_errors=True)
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        self._run(["git", "clone", self._clone_url, str(self.repo_dir)], retries=5)

    def _detect_default_branch(self) -> str:
        """Detect the default branch name from the remote."""
        try:
            result = self._run(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=self.repo_dir,
            )
            # Output: "refs/remotes/origin/main" or "refs/remotes/origin/master"
            return result.stdout.strip().split("/")[-1]
        except subprocess.CalledProcessError:
            pass

        # Fallback: query remote directly
        try:
            result = self._run(
                ["git", "remote", "show", "origin"],
                cwd=self.repo_dir,
            )
            for line in result.stdout.splitlines():
                if "HEAD branch:" in line:
                    return line.split(":")[-1].strip()
        except subprocess.CalledProcessError:
            pass

        logger.warning("Could not detect default branch, falling back to 'main'")
        return "main"

    def ensure_repo(self, branch: str | None = None) -> Path:
        """Clone if not exists, else fetch + reset. Returns repo_dir.

        If *branch* is not specified, the remote's default branch is used.
        """
        if not (self.repo_dir / ".git").exists():
            logger.info("Cloning %s into %s", self.repo_name, self.repo_dir)
            self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run(["git", "clone", self._clone_url, str(self.repo_dir)], retries=5)
        else:
            if branch is None:
                branch = self._detect_default_branch()
            logger.info("Fetching updates for %s (branch=%s)", self.repo_name, branch)
            self._cleanup_lock_files()
            try:
                self._run(["git", "fetch", "origin"], cwd=self.repo_dir, retries=5)
                self._run(["git", "checkout", "--force", branch], cwd=self.repo_dir, retries=5)
                self._run(["git", "reset", "--hard", f"origin/{branch}"], cwd=self.repo_dir, retries=5)
            except subprocess.CalledProcessError:
                logger.warning("Fetch/checkout failed after retries — re-cloning repo")
                self._reclone()
        self._configure_git()
        # Ensure remote URL has the current token
        self._run(["git", "remote", "set-url", "origin", self._clone_url], cwd=self.repo_dir)
        return self.repo_dir

    def create_branch(self, branch_name: str) -> str:
        """Create and checkout a fresh branch from current HEAD.

        If a remote branch with the same name already has an open PR,
        a numeric suffix is appended (e.g. ``issue-14-2``) to avoid
        conflicts. Returns the actual branch name used.
        """
        actual = self._unique_branch_name(branch_name)
        try:
            self._run(["git", "branch", "-D", actual], cwd=self.repo_dir)
        except subprocess.CalledProcessError:
            pass  # branch didn't exist — that's fine
        self._run(["git", "checkout", "-b", actual], cwd=self.repo_dir)
        return actual

    def _unique_branch_name(self, base: str) -> str:
        """Return *base* if the remote branch doesn't exist, else append a suffix."""
        try:
            result = self._run(
                ["git", "ls-remote", "--heads", "origin"],
                cwd=self.repo_dir,
            )
            remote_branches = {
                line.split("refs/heads/")[-1]
                for line in result.stdout.strip().splitlines()
                if "refs/heads/" in line
            }
        except subprocess.CalledProcessError:
            return base

        if base not in remote_branches:
            return base

        attempt = 2
        while f"{base}-{attempt}" in remote_branches:
            attempt += 1
        return f"{base}-{attempt}"

    def has_changes(self) -> bool:
        """Check if working tree has changes."""
        result = self._run(["git", "status", "--porcelain"], cwd=self.repo_dir)
        return bool(result.stdout.strip())

    def commit(self, message: str) -> None:
        """Stage all changes and commit."""
        self._run(["git", "add", "-A"], cwd=self.repo_dir)
        self._run(["git", "commit", "-m", message], cwd=self.repo_dir)

    def push(self, branch: str) -> None:
        """Push branch to origin."""
        self._run(["git", "push", "origin", branch, "--force"], cwd=self.repo_dir, retries=2)
