import logging
from pathlib import Path

import typer
from rich.console import Console

from coding_agent.code_indexer.cli import indexer_app
from coding_agent.rag.cli import rag_app

app = typer.Typer(name="coding-agent", help="Baseline coding agent for GitHub automation.")
app.add_typer(indexer_app, name="index")
app.add_typer(rag_app, name="rag")
console = Console()


def _build_agent(
    work_dir: Path,
    index_dir: Path,
    max_steps: int = 0,
):
    """Create AgenticAgent with file + indexer tools rooted at work_dir."""
    from coding_agent.agents.agentic_agent import AgenticAgent
    from coding_agent.agents.console_callback import ConsoleCallback
    from coding_agent.config import settings
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
        console.print(f"[dim]Code index: {index_dir}[/dim]")

    # RAG tools — look for .rag_index next to work_dir
    rag_index = work_dir / ".rag_index"
    if not rag_index.is_dir():
        rag_index = work_dir.parent / ".rag_index"
    if rag_index.is_dir():
        from coding_agent.rag.config import RAGConfig
        from coding_agent.rag.stores import CodeStore

        _code_count = CodeStore(rag_index, RAGConfig()).count()
        if _code_count > 0:
            from coding_agent.rag.tools import create_rag_tools

            registry.register_many(create_rag_tools(str(rag_index)))
            console.print(f"[dim]RAG index: {rag_index} ({_code_count} chunks)[/dim]")
        else:
            console.print(
                f"[yellow]RAG index dir exists ({rag_index}) but stores are empty — "
                f"run 'coding-agent rag build' first[/yellow]"
            )

    from coding_agent.agents.console_callback import CompositeCallback, MarkdownCallback

    console_cb = ConsoleCallback(console)
    console_cb.print_tools(registry)
    md_cb = MarkdownCallback()
    callback = CompositeCallback([console_cb, md_cb])

    from coding_agent.config import get_model_config

    llm = LLMService(get_model_config("agentic"))
    agent = AgenticAgent(llm=llm, registry=registry, max_steps=steps, callback=callback)
    return agent, md_cb


def _get_git_head(repo_dir: Path) -> str:
    """Return current HEAD commit SHA, or empty string if not a git repo."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _build_index(repo_dir: Path, index_dir: Path, use_llm: bool = False) -> None:
    """Build or incrementally update the code index."""
    from coding_agent.code_indexer.graph.builder import GraphBuilder
    from coding_agent.code_indexer.graph.storage import load_graph, save_graph
    from coding_agent.code_indexer.models import IndexerConfig

    graph_path = index_dir / "graph.json"
    head = _get_git_head(repo_dir)

    if graph_path.exists():
        existing = load_graph(index_dir)
        from coding_agent.code_indexer.models import NodeType

        unsummarized = [n for n in existing.nodes if not n.summary and n.node_type != NodeType.DIRECTORY]
        needs_summaries = use_llm and len(unsummarized) > 0
        has_dirs = any(n.node_type == NodeType.DIRECTORY for n in existing.nodes)
        head_changed = existing.git_head == "" or existing.git_head != head

        # Ensure directory nodes exist even when the rest is up to date
        if not has_dirs:
            console.print("[dim]Building directory nodes...[/dim]")
            _ensure_directory_nodes(existing, repo_dir, use_llm)
            save_graph(existing, index_dir)

        if not head_changed and not needs_summaries:
            console.print(f"[dim]Code index up to date (HEAD {head[:8]})[/dim]")
            return

        # Case 1: HEAD unchanged, just need to fill in missing summaries
        if not head_changed and needs_summaries:
            console.print(
                f"[dim]Adding summaries to {len(unsummarized)} nodes "
                f"(HEAD {head[:8]} unchanged)...[/dim]"
            )
            from coding_agent.code_indexer.summarizer import summarize_modules, summarize_nodes
            from coding_agent.services.llm_service import LLMService

            from coding_agent.config import get_model_config

            llm = LLMService(get_model_config("indexer"))
            summarize_nodes(
                existing.nodes,
                repo_dir,
                llm,
                batch_size=IndexerConfig().batch_size,
            )
            summarize_modules(existing.nodes, existing.edges, llm)
            existing.git_head = head
            summarized = sum(1 for n in existing.nodes if n.summary)
            console.print(
                f"[dim]  Index ready: {len(existing.nodes)} nodes, "
                f"{summarized} summarized[/dim]"
            )
            save_graph(existing, index_dir)
            return

        # Case 2: HEAD changed, do incremental update
        console.print(
            f"[dim]Updating code index (HEAD {existing.git_head[:8] or '?'}"
            f" → {head[:8] or '?'})...[/dim]"
        )
    else:
        existing = None
        console.print("[dim]Building code index (full)...[/dim]")

    kwargs: dict = {}
    if use_llm:
        from coding_agent.services.llm_service import LLMService

        kwargs["config"] = IndexerConfig(use_llm_summaries=True, use_llm_fallback_parser=True)
        from coding_agent.config import get_model_config

        kwargs["llm_service"] = LLMService(get_model_config("indexer"))
    else:
        kwargs["config"] = IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False)

    builder = GraphBuilder(repo_dir, **kwargs)
    if existing is not None:
        graph = builder.update(existing)
    else:
        graph = builder.build()

    graph.git_head = head

    summarized = sum(1 for n in graph.nodes if n.summary)
    console.print(
        f"[dim]  Index ready: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{summarized} summarized[/dim]"
    )
    save_graph(graph, index_dir)


def _build_rag_index(repo_dir: Path, rag_index_dir: Path) -> None:
    """Build or incrementally update the RAG index."""
    from coding_agent.rag.indexer import RAGIndexer

    indexer = RAGIndexer(repo_dir, index_dir=rag_index_dir)
    code_count = indexer._code_store.count()

    if code_count == 0:
        console.print("[dim]Building RAG index (full)...[/dim]")
        stats = indexer.build()
        console.print(
            f"[dim]  RAG index ready: {stats['files']} files, "
            f"{stats['chunks']} chunks, {stats['summaries']} summaries[/dim]"
        )
    else:
        console.print(f"[dim]Updating RAG index ({code_count} existing chunks)...[/dim]")
        stats = indexer.update()
        if "files" in stats:
            # Full rebuild was triggered inside update()
            console.print(
                f"[dim]  RAG index rebuilt: {stats['files']} files, "
                f"{stats['chunks']} chunks, {stats['summaries']} summaries[/dim]"
            )
        else:
            total_changes = stats["added"] + stats["changed"] + stats["deleted"]
            if total_changes == 0:
                console.print(
                    f"[dim]  RAG index up to date ({indexer._code_store.count()} chunks, "
                    f"{indexer._summary_store.count()} summaries)[/dim]"
                )
            else:
                console.print(
                    f"[dim]  RAG index updated: +{stats['added']} added, "
                    f"~{stats['changed']} changed, -{stats['deleted']} deleted, "
                    f"{stats['new_chunks']} new chunks[/dim]"
                )


def _ensure_directory_nodes(graph, repo_dir: Path, use_llm: bool) -> None:
    """Add DIRECTORY nodes (and optionally LLM summaries) to an existing graph."""
    from coding_agent.code_indexer.graph.builder import GraphBuilder
    from coding_agent.code_indexer.models import IndexerConfig

    builder = GraphBuilder(repo_dir, config=IndexerConfig())
    builder._build_directory_nodes(graph.nodes, graph.edges)

    if use_llm:
        try:
            from coding_agent.code_indexer.summarizer import summarize_directories, summarize_modules
            from coding_agent.config import get_model_config
            from coding_agent.services.llm_service import LLMService

            llm = LLMService(get_model_config("indexer"))
            summarize_modules(graph.nodes, graph.edges, llm)
            summarize_directories(graph.nodes, graph.edges, llm)
        except Exception:
            pass  # summaries are optional

    graph._build_indexes()


@app.command()
def solve(
    repo: str = typer.Option(..., help="Repository full name (owner/repo)"),
    issue: int = typer.Option(..., help="Issue number to solve"),
    max_steps: int = typer.Option(0, "--max-steps", help="Max agent steps (0 = use config)"),
    llm_index: bool = typer.Option(False, "--llm-index", help="Use LLM for index summaries and fallback parsing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable detailed logging"),
) -> None:
    """Solve a GitHub issue by creating a PR with code changes."""
    from coding_agent.config import settings
    from coding_agent.models.agent_schemas import MaxStepsError
    from coding_agent.services.github_service import GitHubService
    from coding_agent.services.repo_manager import RepoManager

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s | %(levelname)s | %(message)s")

    token = settings.github_token
    workdir = Path(settings.workdir).expanduser()

    # 1. Clone / pull repo
    rm = RepoManager(repo, token, workdir=workdir)
    repo_dir = rm.ensure_repo()
    console.print(f"[dim]Repo: {repo_dir}[/dim]")

    # 2. Get issue from GitHub
    gh = GitHubService(repo_name=repo)
    issue_obj = gh.get_issue(repo, issue)
    console.print(f"[bold]Issue #{issue}: {issue_obj.title}[/bold]")

    # 3. Create branch
    branch = rm.create_branch(f"issue-{issue}")

    # 4. Build / update index
    _build_index(repo_dir, rm.index_dir, use_llm=llm_index)
    _build_rag_index(repo_dir, rm.rag_index_dir)

    # 5. Run agent
    agent, md_cb = _build_agent(repo_dir, rm.index_dir, max_steps=max_steps)
    task = (
        f"Solve the following GitHub issue.\n\n"
        f"Issue #{issue}: {issue_obj.title}\n\n"
        f"{issue_obj.body}"
    )

    try:
        agent.run(task)
    except MaxStepsError:
        console.print("[red]Agent exceeded max steps.[/red]")
        raise typer.Exit(1)

    # 6. Commit, push, create PR
    if not rm.has_changes():
        console.print("[yellow]No changes made by agent.[/yellow]")
        raise typer.Exit(0)

    rm.commit(f"Solve issue #{issue}: {issue_obj.title}")
    rm.push(branch)

    pr_body = md_cb.build_pr_body(issue, issue_obj.title)
    pr_number = gh.create_pr(
        repo,
        title=f"Fix #{issue}: {issue_obj.title}",
        body=pr_body,
        head=branch,
    )
    try:
        gh.add_label(repo, pr_number, "ai-generated")
    except Exception:
        pass  # label may not exist
    console.print(f"[green]Created PR #{pr_number}[/green]")


@app.command()
def review(
    repo: str = typer.Option(..., help="Repository full name (owner/repo)"),
    pr: int = typer.Option(..., help="PR number to review"),
    max_steps: int = typer.Option(0, "--max-steps", help="Max agent steps (0 = use config)"),
    llm_index: bool = typer.Option(False, "--llm-index", help="Use LLM for index summaries and fallback parsing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable detailed logging"),
) -> None:
    """Review a pull request and post feedback."""
    from coding_agent.config import settings
    from coding_agent.models.agent_schemas import MaxStepsError
    from coding_agent.services.github_service import GitHubService
    from coding_agent.services.repo_manager import RepoManager

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s | %(levelname)s | %(message)s")

    token = settings.github_token
    workdir = Path(settings.workdir).expanduser()

    # 1. Collect context
    gh = GitHubService(repo_name=repo)
    pr_obj = gh.get_pr(repo, pr)
    diff = gh.get_pr_diff(repo, pr)

    # Try to get linked issue
    issue_text = ""
    if pr_obj.body:
        import re
        match = re.search(r"#(\d+)", pr_obj.body)
        if match:
            try:
                linked = gh.get_issue(repo, int(match.group(1)))
                issue_text = f"Linked issue #{linked.number}: {linked.title}\n{linked.body}"
            except Exception:
                pass

    # CI status
    ci_text = ""
    try:
        checks = gh.get_check_runs(repo, pr_obj.head_branch)
        if checks:
            ci_lines = [f"- {c.name}: {c.status}/{c.conclusion}" for c in checks]
            ci_text = "\n".join(ci_lines)
    except Exception:
        pass

    # 2. Ensure repo + index for agent exploration
    rm = RepoManager(repo, token, workdir=workdir)
    repo_dir = rm.ensure_repo()
    _build_index(repo_dir, rm.index_dir, use_llm=llm_index)
    _build_rag_index(repo_dir, rm.rag_index_dir)

    # 3. Run agent
    agent, _ = _build_agent(repo_dir, rm.index_dir, max_steps=max_steps)
    task = (
        f"Review the following pull request and produce a detailed code review.\n\n"
        f"PR #{pr}: {pr_obj.title}\n\n"
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
        "- CI analysis if applicable"
    )

    try:
        result = agent.run(task)
    except MaxStepsError:
        console.print("[red]Agent exceeded max steps.[/red]")
        raise typer.Exit(1)

    # 4. Post review comment
    gh.add_pr_comment(repo, pr, result.output)
    console.print(f"[green]Review posted to PR #{pr}[/green]")


@app.command("agentic-solve")
def agentic_solve(
    task: str = typer.Option(..., help="Task description"),
    repo: str = typer.Option("", help="GitHub repo (owner/repo) for remote mode"),
    file: list[str] = typer.Option([], "-f", "--file", help="Files or directories for context"),
    local: bool = typer.Option(False, "--local", help="Use local file system"),
    index_dir: str = typer.Option("", "--index-dir", help="Path to code index directory"),
    max_steps: int = typer.Option(0, "--max-steps", help="Max agent steps (0 = use config)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write files, only show what would change"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable detailed logging"),
) -> None:
    """Solve a task using an agentic loop with tool calling."""
    from coding_agent.agents.agentic_agent import AgenticAgent
    from coding_agent.agents.console_callback import ConsoleCallback
    from coding_agent.config import settings
    from coding_agent.models.agent_schemas import MaxStepsError
    from coding_agent.services.llm_service import LLMService
    from coding_agent.tools import ToolRegistry
    from coding_agent.tools.base_tools import create_base_tools

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s | %(levelname)s | %(message)s")

    steps = max_steps or settings.agentic_max_steps

    registry = ToolRegistry()

    if local or file:
        from coding_agent.services.local_service import LocalService

        service = LocalService()
        if dry_run:
            service = _dry_run_wrapper(service)
        registry.register_many(create_base_tools(service))
    elif repo:
        from coding_agent.services.github_service import GitHubService

        service = GitHubService(repo_name=repo)
        registry.register_many(create_base_tools(service))
    else:
        console.print("[red]Specify --local/-f for local mode or --repo for GitHub mode.[/red]")
        raise typer.Exit(1)

    resolved_index_dir = index_dir
    if not resolved_index_dir:
        import os

        default = os.path.join(os.getcwd(), ".code_index")
        if os.path.isdir(default):
            resolved_index_dir = default
            console.print(f"[dim]Auto-detected code index at {default}[/dim]")

    if resolved_index_dir:
        from coding_agent.tools.indexer_tools import create_indexer_tools

        registry.register_many(create_indexer_tools(resolved_index_dir))

    # Build full task with file context hints
    full_task = task
    if file:
        file_list = "\n".join(f"  - {f}" for f in file)
        full_task = f"{task}\n\nWork with these files/directories:\n{file_list}"

    callback = ConsoleCallback(console)
    callback.print_tools(registry)

    from coding_agent.config import get_model_config

    llm = LLMService(get_model_config("agentic"))
    agent = AgenticAgent(llm=llm, registry=registry, max_steps=steps, callback=callback)

    console.print(f"[bold]Task:[/bold] {task}")
    if file:
        console.print(f"[bold]Files:[/bold] {', '.join(file)}")
    if dry_run:
        console.print("[yellow]Dry run mode — no files will be written.[/yellow]")
    console.print()

    try:
        agent.run(full_task)

        if dry_run and hasattr(service, "_written"):
            for path, content in service._written.items():
                console.print(f"\n[bold cyan]--- {path} (would write) ---[/bold cyan]")
                console.print(content)

        if repo and not local and not dry_run and hasattr(service, "finalize"):
            service.finalize()
            console.print("[green]Changes committed to GitHub.[/green]")
    except MaxStepsError:
        console.print(f"\n[red]Agent exceeded {steps} steps without completing.[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to listen on"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable detailed logging"),
) -> None:
    """Start the webhook server for GitHub App mode."""
    import uvicorn

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s | %(levelname)s | %(message)s")

    from coding_agent.server import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port)


def _dry_run_wrapper(service):
    """Wrap a FileService so writes/edits are captured but not persisted."""
    from coding_agent.services.file_service import FileService

    class DryRunService(FileService):
        def __init__(self, inner):
            self._inner = inner
            self._written: dict[str, str] = {}

        def read_file(self, path: str) -> str:
            if path in self._written:
                return self._written[path]
            return self._inner.read_file(path)

        def write_file(self, path: str, content: str) -> None:
            self._written[path] = content

        def edit_file(self, path: str, old_text: str, new_text: str) -> None:
            content = self.read_file(path)
            if old_text not in content:
                raise ValueError(f"old_text not found in {path}")
            self._written[path] = content.replace(old_text, new_text, 1)

        def list_directory(self, path: str = ".") -> list[str]:
            return self._inner.list_directory(path)

        def file_exists(self, path: str) -> bool:
            return path in self._written or self._inner.file_exists(path)

    return DryRunService(service)
