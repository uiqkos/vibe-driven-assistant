from __future__ import annotations

import requests
from github import Github, GithubException
from github.GithubRetry import GithubRetry

from coding_agent.config import settings


def _make_github_client(token: str) -> Github:
    """Create a Github client with longer retry backoff for flaky connections."""
    retry = GithubRetry(
        total=6,
        backoff_factor=2,
        backoff_max=60,
    )
    return Github(token, retry=retry, timeout=30)
from coding_agent.services.file_service import FileService
from coding_agent.models.schemas import (
    CheckRun,
    FileChange,
    FileInfo,
    Issue,
    PR,
)


class GitHubService(FileService):
    def __init__(
        self,
        repo_name: str | None = None,
        branch: str | None = None,
        github_client: Github | None = None,
    ) -> None:
        if github_client is not None:
            self.gh = github_client
            self.token = github_client.requester.auth.token if github_client.requester.auth else settings.github_token
        else:
            self.gh = _make_github_client(settings.github_token)
            self.token = settings.github_token
        self.repo_name = repo_name
        if branch is None and repo_name:
            branch = self.gh.get_repo(repo_name).default_branch
        self.branch = branch or "main"
        self._pending_changes: dict[str, str] = {}

    def _repo(self, repo_full_name: str):
        return self.gh.get_repo(repo_full_name)

    # ---- Issues ----

    def get_issue(self, repo_name: str, issue_number: int) -> Issue:
        issue = self._repo(repo_name).get_issue(issue_number)
        return Issue(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            labels=[l.name for l in issue.labels],
        )

    # ---- Repo files ----

    def get_repo_files(self, repo_name: str, ref: str | None = None) -> list[FileInfo]:
        repo = self._repo(repo_name)
        tree = repo.get_git_tree(ref or repo.default_branch, recursive=True)
        files: list[FileInfo] = []
        for item in tree.tree:
            if item.type == "blob" and item.path.endswith(".py"):
                files.append(FileInfo(path=item.path, size=item.size or 0, sha=item.sha))
        files.sort(key=lambda f: f.size)
        return files[: settings.max_context_files]

    def get_file_content(self, repo_name: str, path: str, ref: str | None = None) -> str:
        repo = self._repo(repo_name)
        content_file = repo.get_contents(path, ref=ref or repo.default_branch)
        if isinstance(content_file, list):
            return ""
        if (content_file.size or 0) > settings.max_file_size_kb * 1024:
            return ""
        return content_file.decoded_content.decode()

    # ---- Branches & commits ----

    def create_branch(self, repo_name: str, branch_name: str, base: str | None = None) -> None:
        repo = self._repo(repo_name)
        base_sha = repo.get_branch(base or repo.default_branch).commit.sha
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)

    def commit_files(
        self, repo_name: str, branch: str, files: list[FileChange], message: str
    ) -> None:
        repo = self._repo(repo_name)
        for f in files:
            try:
                existing = repo.get_contents(f.path, ref=branch)
                if isinstance(existing, list):
                    continue
                repo.update_file(f.path, message, f.content, existing.sha, branch=branch)
            except GithubException:
                repo.create_file(f.path, message, f.content, branch=branch)

    # ---- Pull Requests ----

    def create_pr(
        self, repo_name: str, title: str, body: str, head: str, base: str | None = None
    ) -> int:
        repo = self._repo(repo_name)
        if base is None:
            base = repo.default_branch
        pr = repo.create_pull(title=title, body=body, head=head, base=base)
        return pr.number

    def get_pr(self, repo_name: str, pr_number: int) -> PR:
        pr = self._repo(repo_name).get_pull(pr_number)
        return PR(
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            head_branch=pr.head.ref,
            base_branch=pr.base.ref,
            labels=[l.name for l in pr.labels],
        )

    def get_pr_diff(self, repo_name: str, pr_number: int) -> str:
        url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3.diff",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text

    def add_pr_comment(self, repo_name: str, pr_number: int, body: str) -> None:
        self._repo(repo_name).get_issue(pr_number).create_comment(body)

    def get_pr_comments(self, repo_name: str, pr_number: int) -> list[str]:
        comments = self._repo(repo_name).get_issue(pr_number).get_comments()
        return [c.body for c in comments]

    def get_check_runs(self, repo_name: str, ref: str) -> list[CheckRun]:
        repo = self._repo(repo_name)
        commit = repo.get_commit(ref)
        runs: list[CheckRun] = []
        for cr in commit.get_check_runs():
            runs.append(
                CheckRun(
                    name=cr.name,
                    status=cr.status,
                    conclusion=cr.conclusion,
                    output_title=cr.output.title if cr.output else None,
                    output_summary=cr.output.summary if cr.output else None,
                )
            )
        return runs

    def add_label(self, repo_name: str, issue_number: int, label: str) -> None:
        self._repo(repo_name).get_issue(issue_number).add_to_labels(label)

    # --- FileService interface ---

    def read_file(self, path: str) -> str:
        if path in self._pending_changes:
            return self._pending_changes[path]
        assert self.repo_name, "repo_name required for FileService methods"
        return self.get_file_content(self.repo_name, path, ref=self.branch)

    def write_file(self, path: str, content: str) -> None:
        self._pending_changes[path] = content

    def edit_file(self, path: str, old_text: str, new_text: str) -> None:
        content = self.read_file(path)
        if old_text not in content:
            raise ValueError(f"old_text not found in {path}")
        self._pending_changes[path] = content.replace(old_text, new_text, 1)

    def list_directory(self, path: str = ".") -> list[str]:
        assert self.repo_name, "repo_name required for FileService methods"
        repo = self._repo(self.repo_name)
        try:
            contents = repo.get_contents(path if path != "." else "", ref=self.branch)
        except Exception:
            return []
        if not isinstance(contents, list):
            return [contents.path]
        return sorted(item.path for item in contents)

    def file_exists(self, path: str) -> bool:
        if path in self._pending_changes:
            return True
        assert self.repo_name, "repo_name required for FileService methods"
        try:
            self._repo(self.repo_name).get_contents(path, ref=self.branch)
            return True
        except Exception:
            return False

    def finalize(self, message: str = "Apply agent changes") -> None:
        if not self._pending_changes or not self.repo_name:
            return
        files = [FileChange(path=p, content=c) for p, c in self._pending_changes.items()]
        self.commit_files(self.repo_name, self.branch, files, message)
        self._pending_changes.clear()
