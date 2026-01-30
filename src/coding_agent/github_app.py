from __future__ import annotations

from urllib3.util.retry import Retry

from github import Github, GithubIntegration

from coding_agent.config import settings

_RETRY = Retry(
    total=6,
    backoff_factor=2,
    backoff_max=60,
    status_forcelist=[502, 503, 504],
    allowed_methods=None,
    raise_on_status=False,
)


def _make_github(token: str) -> Github:
    """Create a PyGithub client with automatic retries."""
    return Github(token, retry=_RETRY)


class GitHubAppAuth:
    """Authenticate as a GitHub App installation and return a PyGithub client."""

    def __init__(self) -> None:
        if not settings.github_app_id or not settings.github_app_private_key_path:
            raise RuntimeError(
                "GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY_PATH must be set for GitHub App mode."
            )
        with open(settings.github_app_private_key_path) as f:
            self._private_key = f.read()
        self._app_id = settings.github_app_id
        self._integration = GithubIntegration(
            integration_id=self._app_id,
            private_key=self._private_key,
            retry=_RETRY,
        )

    def get_installation_token(self, installation_id: int) -> str:
        """Get an installation access token for a specific installation."""
        return self._integration.get_access_token(installation_id).token

    def get_github_client(self, installation_id: int) -> Github:
        """Return a PyGithub client authenticated as the given installation."""
        token = self.get_installation_token(installation_id)
        return _make_github(token)
