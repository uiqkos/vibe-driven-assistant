"""Typer sub-app for code indexer commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from coding_agent.code_indexer.graph.builder import GraphBuilder
from coding_agent.code_indexer.graph.query import GraphQuery
from coding_agent.code_indexer.graph.storage import load_graph, save_graph
from coding_agent.code_indexer.models import IndexerConfig

indexer_app = typer.Typer(name="index", help="Code indexer commands.")
console = Console()


def _get_llm_service():
    try:
        from coding_agent.config import get_model_config
        from coding_agent.services.llm_service import LLMService
        return LLMService(get_model_config("indexer"))
    except Exception:
        return None


@indexer_app.command()
def index(
    project: Path = typer.Option(".", help="Project root directory"),
    output: Path = typer.Option(None, help="Output directory for index"),
    no_llm: bool = typer.Option(False, help="Disable LLM summaries and fallback parsing"),
) -> None:
    """Build a full index of the project."""
    config = IndexerConfig(use_llm_summaries=not no_llm, use_llm_fallback_parser=not no_llm)
    llm = None if no_llm else _get_llm_service()
    builder = GraphBuilder(project, config=config, llm_service=llm)

    console.print(f"[bold]Indexing {project}...[/bold]")
    graph = builder.build()
    path = save_graph(graph, output)
    console.print(
        f"[green]Index created: {path}[/green]\n"
        f"  Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}, Files: {len(graph.checksums)}"
    )


@indexer_app.command()
def update(
    project: Path = typer.Option(".", help="Project root directory"),
    index_dir: Path = typer.Option(None, help="Index directory"),
) -> None:
    """Incrementally update the index for changed files."""
    idx = index_dir or (project / ".code_index")
    existing = load_graph(idx)
    llm = _get_llm_service()
    builder = GraphBuilder(project, llm_service=llm)

    console.print("[bold]Updating index...[/bold]")
    graph = builder.update(existing)
    save_graph(graph, idx)
    console.print(
        f"[green]Index updated.[/green]\n"
        f"  Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}, Files: {len(graph.checksums)}"
    )


@indexer_app.command()
def show(
    file: str = typer.Option(..., help="File path to show structure of"),
    index_dir: Path = typer.Option(".code_index", help="Index directory"),
) -> None:
    """Show the structure of an indexed file."""
    graph = load_graph(index_dir)
    q = GraphQuery(graph)
    nodes = q.get_file_structure(file)
    if not nodes:
        console.print(f"[yellow]No nodes found for '{file}'[/yellow]")
        return
    console.print(f"[bold]{file}[/bold]")
    for n in nodes:
        indent = "  " if n.node_type.value in ("method",) else ""
        sig = f"  {n.signature}" if n.signature else ""
        console.print(f"{indent}[cyan]{n.node_type.value:8}[/cyan] {n.name} (L{n.line_start}-{n.line_end}){sig}")


@indexer_app.command()
def search(
    query: str = typer.Option(..., help="Search query"),
    index_dir: Path = typer.Option(".code_index", help="Index directory"),
) -> None:
    """Search the index by name, docstring, or summary."""
    graph = load_graph(index_dir)
    q = GraphQuery(graph)
    results = q.search(query)
    if not results:
        console.print(f"[yellow]No results for '{query}'[/yellow]")
        return
    console.print(f"[bold]{len(results)} results for '{query}':[/bold]")
    for n in results[:30]:
        console.print(f"  [cyan]{n.node_type.value:8}[/cyan] {n.id}")


@indexer_app.command()
def deps(
    node_id: str = typer.Option(..., help="Node ID to show dependencies for"),
    index_dir: Path = typer.Option(".code_index", help="Index directory"),
) -> None:
    """Show dependencies and dependents of a node."""
    graph = load_graph(index_dir)
    q = GraphQuery(graph)
    console.print(f"[bold]Dependencies of {node_id}:[/bold]")
    for n in q.get_dependencies(node_id):
        console.print(f"  → {n.id}")
    console.print(f"\n[bold]Dependents of {node_id}:[/bold]")
    for n in q.get_dependents(node_id):
        console.print(f"  ← {n.id}")


@indexer_app.command()
def stats(
    index_dir: Path = typer.Option(".code_index", help="Index directory"),
) -> None:
    """Show statistics about the index."""
    graph = load_graph(index_dir)
    from collections import Counter
    type_counts = Counter(n.node_type.value for n in graph.nodes)
    edge_counts = Counter(e.edge_type.value for e in graph.edges)

    console.print("[bold]Index Statistics[/bold]")
    console.print(f"  Files: {len(graph.checksums)}")
    console.print(f"  Nodes: {len(graph.nodes)}")
    for t, c in sorted(type_counts.items()):
        console.print(f"    {t}: {c}")
    console.print(f"  Edges: {len(graph.edges)}")
    for t, c in sorted(edge_counts.items()):
        console.print(f"    {t}: {c}")
    summarized = sum(1 for n in graph.nodes if n.summary)
    console.print(f"  Nodes with summaries: {summarized}")
