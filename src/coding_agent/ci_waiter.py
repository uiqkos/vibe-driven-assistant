from __future__ import annotations

import asyncio
import logging

from coding_agent.models.schemas import CheckRun
from coding_agent.services.github_service import GitHubService

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 30
DEFAULT_TIMEOUT = 600


class CIWaiter:
    def __init__(self, gh: GitHubService) -> None:
        self.gh = gh

    async def wait_for_ci(
        self,
        repo: str,
        ref: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> list[CheckRun]:
        """Poll check_runs until all are completed or timeout is reached."""
        elapsed = 0
        while elapsed < timeout:
            checks = self.gh.get_check_runs(repo, ref)
            if not checks:
                logger.info("No CI checks found for %s, skipping wait", ref)
                return []
            if all(c.status == "completed" for c in checks):
                logger.info("CI completed for %s (%d checks)", ref, len(checks))
                return checks
            completed = sum(1 for c in checks if c.status == "completed")
            logger.info(
                "CI pending for %s (%d/%d completed), waiting %ds...",
                ref,
                completed,
                len(checks),
                poll_interval,
            )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        # Return whatever we have on timeout
        logger.warning("CI timeout (%ds) for %s", timeout, ref)
        return self.gh.get_check_runs(repo, ref)

    @staticmethod
    def ci_passed(checks: list[CheckRun]) -> bool:
        """Return True if all completed checks have a successful conclusion."""
        if not checks:
            return True  # No checks = nothing failed
        return all(
            c.conclusion in ("success", "neutral", "skipped")
            for c in checks
            if c.status == "completed"
        )
