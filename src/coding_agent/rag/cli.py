"""Typer sub-app for RAG commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

rag_app = typer.Typer(name="rag", help="RAG index commands (ChromaDB-based semantic search).")
console = Console()


@rag_app.command()
def build(
    project: Path = typer.Option(".", help="Project root directory"),
    index_dir: Path = typer.Option(None, help="RAG index directory (default: <project>/.rag_index)"),
) -> None:
    """Build the RAG index from scratch."""
    from coding_agent.rag.indexer import RAGIndexer

    indexer = RAGIndexer(project, index_dir=index_dir)
    console.print(f"[bold]Building RAG index for {project}...[/bold]")
    stats = indexer.build()
    console.print(
        f"[green]RAG index built:[/green] "
        f"{stats['files']} files, {stats['chunks']} chunks, {stats['summaries']} summaries"
    )
    console.print(
        f"[dim]Final stores: {indexer._code_store.count()} code chunks, "
        f"{indexer._summary_store.count()} summaries[/dim]"
    )


@rag_app.command()
def update(
    project: Path = typer.Option(".", help="Project root directory"),
    index_dir: Path = typer.Option(None, help="RAG index directory"),
) -> None:
    """Incrementally update the RAG index."""
    from coding_agent.rag.indexer import RAGIndexer

    indexer = RAGIndexer(project, index_dir=index_dir)

    code_count = indexer._code_store.count()
    summary_count = indexer._summary_store.count()
    console.print(f"[dim]Current stores: {code_count} code chunks, {summary_count} summaries[/dim]")

    if code_count == 0:
        console.print("[yellow]Stores are empty — performing full build instead of update[/yellow]")

    console.print("[bold]Updating RAG index...[/bold]")
    stats = indexer.update()

    if "files" in stats:
        # Full build was triggered
        console.print(
            f"[green]RAG index built (full):[/green] "
            f"{stats['files']} files, {stats['chunks']} chunks, {stats['summaries']} summaries"
        )
    else:
        console.print(
            f"[green]RAG index updated:[/green] "
            f"+{stats['added']} added, ~{stats['changed']} changed, -{stats['deleted']} deleted, "
            f"{stats['new_chunks']} new chunks, {stats['summaries']} summaries"
        )

    console.print(
        f"[dim]Final stores: {indexer._code_store.count()} code chunks, "
        f"{indexer._summary_store.count()} summaries[/dim]"
    )


@rag_app.command()
def search(
    query: str = typer.Option(..., help="Search query"),
    store: str = typer.Option("hybrid", help="Store to search: code, summary, or hybrid"),
    index_dir: Path = typer.Option(".rag_index", help="RAG index directory"),
    top_k: int = typer.Option(10, help="Number of results"),
) -> None:
    """Search the RAG index."""
    from coding_agent.rag.tools import _search_code, _search_hybrid, _search_semantic

    idx = str(index_dir)
    if store == "code":
        result = _search_code(idx, query, top_k)
    elif store == "summary":
        result = _search_semantic(idx, query, top_k)
    else:
        result = _search_hybrid(idx, query, top_k)

    console.print(result)


@rag_app.command()
def stats(
    index_dir: Path = typer.Option(".rag_index", help="RAG index directory"),
) -> None:
    """Show RAG index statistics."""
    from coding_agent.rag.config import RAGConfig
    from coding_agent.rag.stores import CodeStore, SummaryStore

    config = RAGConfig()
    code = CodeStore(index_dir, config)
    summary = SummaryStore(index_dir, config)

    console.print("[bold]RAG Index Statistics[/bold]")
    console.print(f"  Code chunks: {code.count()}")
    console.print(f"  Summaries: {summary.count()}")
