from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class Issue(BaseModel):
    number: int
    title: str
    body: str
    labels: list[str] = []


class PR(BaseModel):
    number: int
    title: str
    body: str
    head_branch: str
    base_branch: str = "main"
    labels: list[str] = []


class FileInfo(BaseModel):
    path: str
    size: int
    sha: str


class FileChange(BaseModel):
    path: str
    content: str


class CheckRun(BaseModel):
    name: str
    status: str
    conclusion: str | None = None
    output_title: str | None = None
    output_summary: str | None = None


class FileAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class CodeSolution(BaseModel):
    files: list[FileChange]
    explanation: str


class ReviewIssue(BaseModel):
    file: str
    line: int | None = None
    severity: str
    description: str


class ReviewStatus(str, Enum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"


class ReviewResult(BaseModel):
    status: ReviewStatus
    summary: str
    issues: list[ReviewIssue] = []
    ci_analysis: str = ""
