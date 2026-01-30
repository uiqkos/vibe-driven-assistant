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
            except subprocess.CalledProcessError:
                if attempt < retries:
                    delay = 2 ** attempt
                    logger.warning("Command failed (attempt %d/%d), retrying in %ds: %s",
                                   attempt + 1, retries + 1, delay, " ".join(cmd))
                    time.sleep(delay)
                else:
                    raise

    def _configure_git(self) -> None:
        """Set git user identity if not already configured."""
        try:
            self._run(["git", "config", "user.name"], cwd=self.repo_dir)
        except subprocess.CalledProcessError:
            self._run(["git", "config", "user.name", "coding-agent"], cwd=self.repo_dir)
            self._run(["git", "config", "user.email", "coding-agent@noreply"], cwd=self.repo_dir)

    def ensure_repo(self, branch: str = "main") -> Path:
        """Clone if not exists, else fetch + reset. Returns repo_dir."""
        if not (self.repo_dir / ".git").exists():
            logger.info("Cloning %s into %s", self.repo_name, self.repo_dir)
            self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run(["git", "clone", self._clone_url, str(self.repo_dir)], retries=2)
        else:
            logger.info("Fetching updates for %s", self.repo_name)
            self._run(["git", "fetch", "origin"], cwd=self.repo_dir, retries=2)
            self._run(["git", "checkout", "--force", branch], cwd=self.repo_dir)
            self._run(["git", "reset", "--hard", f"origin/{branch}"], cwd=self.repo_dir)
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
