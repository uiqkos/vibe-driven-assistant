from coding_agent.models.schemas import (
    CodeSolution,
    FileChange,
    Issue,
    ReviewIssue,
    ReviewResult,
    ReviewStatus,
)
from coding_agent.config import Settings


def test_issue_model():
    issue = Issue(number=1, title="Bug", body="Fix it")
    assert issue.number == 1
    assert issue.labels == []


def test_code_solution_model():
    sol = CodeSolution(
        files=[FileChange(path="main.py", content="print('hello')")],
        explanation="Added greeting",
    )
    assert len(sol.files) == 1
    assert sol.files[0].path == "main.py"


def test_review_result_model():
    result = ReviewResult(
        status=ReviewStatus.APPROVED,
        summary="Looks good",
        issues=[ReviewIssue(file="main.py", severity="info", description="Minor style")],
    )
    assert result.status == ReviewStatus.APPROVED
    assert len(result.issues) == 1


def test_settings_defaults():
    s = Settings(github_token="t", llm_api_key="k")
    assert s.max_context_files == 10
    assert s.max_iterations == 3
