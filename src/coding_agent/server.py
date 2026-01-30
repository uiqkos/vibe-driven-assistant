from __future__ import annotations

import hashlib
import hmac
import logging
import re
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from coding_agent.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="coding-agent webhook server")


def _verify_signature(payload: bytes, signature: str | None) -> None:
    """Verify GitHub webhook X-Hub-Signature-256."""
    secret = settings.github_app_webhook_secret
    if not secret:
        return
    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature")
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")


def _get_installation_token(installation_id: int) -> str:
    """Get a raw token string for an installation (used by RepoManager + GitHubService)."""
    from coding_agent.github_app import GitHubAppAuth

    auth = GitHubAppAuth()
    return auth.get_installation_token(installation_id)


def _get_gh_service(token: str, repo_name: str):
    from coding_agent.github_app import _make_github
    from coding_agent.services.github_service import GitHubService

    client = _make_github(token)
    return GitHubService(repo_name=repo_name, github_client=client)


# ---------------------------------------------------------------------------
# Shared agentic pipeline (same as CLI solve/review)
# ---------------------------------------------------------------------------

def _build_agent_headless(work_dir: Path, index_dir: Path, max_steps: int = 0):
    """Build an AgenticAgent without console output (for server use).

    Returns (agent, md_callback).
    """
    from coding_agent.agents.agentic_agent import AgenticAgent
    from coding_agent.agents.console_callback import MarkdownCallback
    from coding_agent.config import get_model_config
    from coding_agent.services.llm_service import LLMService
    from coding_agent.services.local_service import LocalService
    from coding_agent.tools import ToolRegistry
    from coding_agent.tools.base_tools import create_base_tools
    from coding_agent.tools.indexer_tools import create_indexer_tools

    steps = max_steps or settings.agentic_max_steps
    registry = ToolRegistry()

    service = LocalService(work_dir=work_dir)
    registry.register_many(create_base_tools(service))

    if index_dir.is_dir():
        registry.register_many(create_indexer_tools(str(index_dir)))

    # RAG tools
    rag_index = work_dir / ".rag_index"
    if not rag_index.is_dir():
        rag_index = work_dir.parent / ".rag_index"
    if rag_index.is_dir():
        from coding_agent.rag.config import RAGConfig
        from coding_agent.rag.stores import CodeStore

        if CodeStore(rag_index, RAGConfig()).count() > 0:
            from coding_agent.rag.tools import create_rag_tools

            registry.register_many(create_rag_tools(str(rag_index)))
            logger.info("RAG tools registered from %s", rag_index)

    md_cb = MarkdownCallback()
    llm = LLMService(get_model_config("agentic"))
    agent = AgenticAgent(llm=llm, registry=registry, max_steps=steps, callback=md_cb)
    return agent, md_cb


def _build_index_headless(repo_dir: Path, index_dir: Path) -> None:
    """Build/update code index without console output.

    1. If graph exists, use MD5 per-file checks to update changed files only.
    2. Backfill missing summaries for any nodes that lack them.
    3. Persist the graph.
    """
    import subprocess

    from coding_agent.code_indexer.graph.builder import GraphBuilder
    from coding_agent.code_indexer.graph.storage import load_graph, save_graph
    from coding_agent.code_indexer.models import IndexerConfig
    from coding_agent.config import get_model_config
    from coding_agent.services.llm_service import LLMService

    graph_path = index_dir / "graph.json"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        )
        head = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = ""

    config = IndexerConfig(use_llm_summaries=True, use_llm_fallback_parser=True)
    llm_service = LLMService(get_model_config("indexer"))

    if graph_path.exists():
        existing = load_graph(index_dir)
        head_changed = existing.git_head == "" or existing.git_head != head

        if head_changed:
            logger.info("Updating code index (HEAD %s → %s)",
                        existing.git_head[:8] or "?", head[:8] or "?")
            builder = GraphBuilder(repo_dir, config=config, llm_service=llm_service)
            graph = builder.update(existing)
            graph.git_head = head
        else:
            logger.debug("Code index up to date (HEAD %s)", head[:8])
            graph = existing
    else:
        logger.info("Building code index (full)")
        builder = GraphBuilder(repo_dir, config=config, llm_service=llm_service)
        graph = builder.build()
        graph.git_head = head

    # Backfill missing summaries (covers previously built graphs without LLM)
    _ensure_summaries(graph, repo_dir, llm_service, config)

    logger.info("Index ready: %d nodes, %d edges", len(graph.nodes), len(graph.edges))
    save_graph(graph, index_dir)


def _ensure_summaries(graph, repo_dir: Path, llm_service, config) -> None:  # noqa: ANN001
    """Generate summaries for any nodes that are missing them."""
    from coding_agent.code_indexer.summarizer import (
        summarize_directories,
        summarize_modules,
        summarize_nodes,
    )

    unsummarized = sum(1 for n in graph.nodes if not n.summary)
    if unsummarized == 0:
        logger.debug("All nodes already have summaries")
        return

    logger.info("Backfilling %d missing summaries", unsummarized)
    summarize_nodes(graph.nodes, repo_dir, llm_service, batch_size=config.batch_size)
    summarize_modules(graph.nodes, graph.edges, llm_service)
    summarize_directories(graph.nodes, graph.edges, llm_service)


def _sync_rag_index(repo_dir: Path, rag_index_dir: Path) -> None:
    """Build or incrementally update the RAG vector index."""
    from coding_agent.rag.indexer import RAGIndexer

    indexer = RAGIndexer(repo_dir, index_dir=rag_index_dir)

    if indexer._code_store.count() == 0:
        logger.info("RAG index empty — full build")
        indexer.build()
    else:
        logger.info("RAG index exists — incremental update")
        indexer.update()


# ---------------------------------------------------------------------------
# Event handlers (run as background tasks)
# ---------------------------------------------------------------------------

def _handle_issue_labeled(payload: dict) -> None:
    """Handle issues.labeled — run agentic solve (same as CLI `coding-agent solve`)."""
    label = payload.get("label", {}).get("name", "")
    if label != "ai-solve":
        return

    repo_name = payload["repository"]["full_name"]
    issue_number = payload["issue"]["number"]
    installation_id = payload["installation"]["id"]

    logger.info("Solving issue #%d in %s (agentic)", issue_number, repo_name)

    token = _get_installation_token(installation_id)
    gh = _get_gh_service(token, repo_name)

    from coding_agent.services.repo_manager import RepoManager

    workdir = Path(settings.workdir).expanduser()

    # 1. Clone / pull repo
    rm = RepoManager(repo_name, token, workdir=workdir)
    repo_dir = rm.ensure_repo()
    logger.info("Repo cloned/updated: %s", repo_dir)

    # 2. Get issue
    issue_obj = gh.get_issue(repo_name, issue_number)
    logger.info("Issue #%d: %s", issue_number, issue_obj.title)

    # 3. Create branch
    branch = rm.create_branch(f"issue-{issue_number}")

    # 4. Build index + RAG
    _build_index_headless(repo_dir, rm.index_dir)
    _sync_rag_index(repo_dir, rm.rag_index_dir)

    # 5. Run agentic agent
    agent, md_cb = _build_agent_headless(repo_dir, rm.index_dir)
    task = (
        f"Solve the following GitHub issue.\n\n"
        f"Issue #{issue_number}: {issue_obj.title}\n\n"
        f"{issue_obj.body}"
    )

    try:
        result = agent.run(task)
        logger.info("Agent finished in %d steps, %d tool calls", result.steps, result.tool_calls_made)
    except Exception:
        logger.exception("Agent failed for issue #%d", issue_number)
        return

    # 6. Commit, push, create PR
    if not rm.has_changes():
        logger.warning("No changes made by agent for issue #%d", issue_number)
        return

    rm.commit(f"Solve issue #{issue_number}: {issue_obj.title}")
    rm.push(branch)

    pr_body = md_cb.build_pr_body(issue_number, issue_obj.title)
    pr_number = gh.create_pr(
        repo_name,
        title=f"Fix #{issue_number}: {issue_obj.title}",
        body=pr_body,
        head=branch,
    )
    try:
        gh.add_label(repo_name, pr_number, "ai-generated")
    except Exception:
        pass
    logger.info("Created PR #%d for issue #%d", pr_number, issue_number)


def _handle_pr_event(payload: dict) -> None:
    """Handle pull_request events — wait for CI then run agentic review (same as CLI `coding-agent review`)."""
    import asyncio

    pr = payload["pull_request"]
    labels = [l["name"] for l in pr.get("labels", [])]
    if "ai-generated" not in labels:
        logger.info("PR #%s skipped — no 'ai-generated' label (labels=%s)", pr.get("number"), labels)
        return

    repo_name = payload["repository"]["full_name"]
    pr_number = pr["number"]
    head_sha = pr["head"]["sha"]
    installation_id = payload["installation"]["id"]

    logger.info("PR #%d: starting review pipeline (sha=%s)", pr_number, head_sha[:8])

    token = _get_installation_token(installation_id)
    gh = _get_gh_service(token, repo_name)

    # Wait for CI
    from coding_agent.ci_waiter import CIWaiter

    waiter = CIWaiter(gh)
    checks = asyncio.run(waiter.wait_for_ci(repo_name, head_sha))

    ci_text = ""
    if checks:
        ci_lines = [f"- {c.name}: {c.status}/{c.conclusion}" for c in checks]
        ci_text = "\n".join(ci_lines)

    # Collect context
    diff = gh.get_pr_diff(repo_name, pr_number)
    pr_obj = gh.get_pr(repo_name, pr_number)

    issue_text = ""
    if pr_obj.body:
        m = re.search(r"#(\d+)", pr_obj.body)
        if m:
            try:
                linked = gh.get_issue(repo_name, int(m.group(1)))
                issue_text = f"Linked issue #{linked.number}: {linked.title}\n{linked.body}"
            except Exception:
                pass

    # Clone/pull repo + build index for agent
    from coding_agent.services.repo_manager import RepoManager

    workdir = Path(settings.workdir).expanduser()
    rm = RepoManager(repo_name, token, workdir=workdir)
    repo_dir = rm.ensure_repo()
    _build_index_headless(repo_dir, rm.index_dir)
    _sync_rag_index(repo_dir, rm.rag_index_dir)

    # Run agentic review
    agent, _ = _build_agent_headless(repo_dir, rm.index_dir)
    task = (
        f"Review the following pull request and produce a detailed code review.\n\n"
        f"PR #{pr_number}: {pr_obj.title}\n\n"
        f"{pr_obj.body or ''}\n\n"
        f"Diff:\n```\n{diff}\n```\n\n"
    )
    if issue_text:
        task += f"Linked issue:\n{issue_text}\n\n"
    if ci_text:
        task += f"CI status:\n{ci_text}\n\n"
    task += (
        "Produce your review as a structured comment with:\n"
        "- Overall assessment (approve / changes requested)\n"
        "- Summary of findings\n"
        "- Specific issues with file paths and line numbers\n"
        "- CI analysis if applicable\n\n"
        "Start your review with the marker: AI Code Review"
    )

    logger.info("PR #%d: CI done (%s), running agentic review...", pr_number, ci_text or "no checks")

    try:
        result = agent.run(task)
        logger.info("PR #%d: review agent finished in %d steps", pr_number, result.steps)
    except Exception:
        logger.exception("Review agent failed for PR #%d", pr_number)
        return

    gh.add_pr_comment(repo_name, pr_number, result.output)
    logger.info("PR #%d: review posted", pr_number)


REVIEW_MARKER = "AI Code Review"


def _handle_issue_comment(payload: dict) -> None:
    """Handle issue_comment.created — run agentic iterate on the PR."""
    comment_body = payload.get("comment", {}).get("body", "")
    if "changes requested" not in comment_body.lower():
        return
    if REVIEW_MARKER not in comment_body:
        return

    issue = payload["issue"]
    if "pull_request" not in issue:
        return

    repo_name = payload["repository"]["full_name"]
    pr_number = issue["number"]
    installation_id = payload["installation"]["id"]

    logger.info("Iterating on PR #%d in %s (agentic)", pr_number, repo_name)

    token = _get_installation_token(installation_id)
    gh = _get_gh_service(token, repo_name)

    # Check iteration count
    comments = gh.get_pr_comments(repo_name, pr_number)
    iteration_count = sum(1 for c in comments if REVIEW_MARKER in c)
    if iteration_count > settings.max_iterations:
        gh.add_pr_comment(
            repo_name, pr_number,
            f"⛔ Max iterations ({settings.max_iterations}) reached.",
        )
        return

    # Get PR details
    pr_obj = gh.get_pr(repo_name, pr_number)

    # Get latest review feedback
    feedback = ""
    for c in reversed(comments):
        if REVIEW_MARKER in c:
            feedback = c
            break

    if not feedback:
        logger.info("PR #%d: no review feedback found, skipping", pr_number)
        return

    # Clone/pull repo + checkout PR branch + build index
    from coding_agent.services.repo_manager import RepoManager

    workdir = Path(settings.workdir).expanduser()
    rm = RepoManager(repo_name, token, workdir=workdir)
    repo_dir = rm.ensure_repo(branch=pr_obj.head_branch)
    _build_index_headless(repo_dir, rm.index_dir)
    _sync_rag_index(repo_dir, rm.rag_index_dir)

    # Run agentic iterate
    agent, _ = _build_agent_headless(repo_dir, rm.index_dir)

    # Get linked issue context
    issue_context = ""
    m = re.search(r"#(\d+)", pr_obj.body)
    if m:
        try:
            linked = gh.get_issue(repo_name, int(m.group(1)))
            issue_context = f"\n\nOriginal issue #{linked.number}: {linked.title}\n{linked.body}"
        except Exception:
            pass

    task = (
        f"You are iterating on PR #{pr_number} based on review feedback.\n\n"
        f"Apply the requested changes to the codebase.\n\n"
        f"Review feedback:\n{feedback}"
        f"{issue_context}"
    )

    try:
        result = agent.run(task)
        logger.info("PR #%d: iterate agent finished in %d steps", pr_number, result.steps)
    except Exception:
        logger.exception("Iterate agent failed for PR #%d", pr_number)
        return

    if not rm.has_changes():
        logger.warning("PR #%d: no changes made by iterate agent", pr_number)
        gh.add_pr_comment(repo_name, pr_number, "No changes produced by iteration.")
        return

    rm.commit(f"Iterate on review feedback (#{pr_number})")
    rm.push(pr_obj.head_branch)
    logger.info("PR #%d: iteration pushed", pr_number)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(None),
    x_github_event: str | None = Header(None),
):
    payload_bytes = await request.body()
    _verify_signature(payload_bytes, x_hub_signature_256)

    payload = await request.json()
    action = payload.get("action", "")
    event = x_github_event or ""

    repo_name = payload.get("repository", {}).get("full_name", "?")
    logger.info("Received event=%s action=%s repo=%s", event, action, repo_name)

    if event == "issues" and action == "labeled":
        label = payload.get("label", {}).get("name", "")
        logger.info("Issue #%s labeled '%s'", payload.get("issue", {}).get("number"), label)
        background_tasks.add_task(_handle_issue_labeled, payload)
    elif event == "pull_request" and action in ("opened", "synchronize", "labeled"):
        pr = payload.get("pull_request", {})
        labels = [l["name"] for l in pr.get("labels", [])]
        logger.info("PR #%s action=%s labels=%s", pr.get("number"), action, labels)
        background_tasks.add_task(_handle_pr_event, payload)
    elif event == "issue_comment" and action == "created":
        body_preview = payload.get("comment", {}).get("body", "")[:80]
        logger.info("Comment on #%s: %s...", payload.get("issue", {}).get("number"), body_preview)
        background_tasks.add_task(_handle_issue_comment, payload)
    else:
        logger.debug("Ignoring event=%s action=%s", event, action)

    return {"status": "accepted"}
