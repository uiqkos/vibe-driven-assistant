"""Agent tools for querying the code index."""

from __future__ import annotations

import logging
from pathlib import Path

from coding_agent.code_indexer.graph.query import GraphQuery
from coding_agent.code_indexer.graph.storage import load_graph, save_graph
from coding_agent.code_indexer.models import EdgeType, NodeType

logger = logging.getLogger(__name__)


def _get_query(index_dir: str = ".code_index") -> GraphQuery:
    graph = load_graph(Path(index_dir))
    _fixup_root_path(graph)
    return GraphQuery(graph)


def _fixup_root_path(graph) -> None:
    """Rewrite root_path when running in a different environment (e.g. Docker).

    The graph stores an absolute root_path from the machine that built
    the index (e.g. ``/Users/kirill/.coding-agent/repos/…``).  When the
    same ``~/.coding-agent`` tree is mounted into a Docker container at
    ``/root/.coding-agent/…`` the original path doesn't exist.

    Strategy: detect a ``/.coding-agent/`` segment inside root_path and
    re-anchor it to the current ``$HOME``.  Fallback: infer root from
    ``index_dir`` location.
    """
    stored = Path(graph.root_path)
    if stored.is_dir():
        return

    marker = "/.coding-agent/"
    root_str: str = graph.root_path
    idx = root_str.find(marker)
    if idx != -1:
        suffix = root_str[idx:]  # e.g. /.coding-agent/repos/owner/repo/source
        new_root = str(Path.home()) + suffix
        if Path(new_root).is_dir():
            logger.info("Remapped root_path %s → %s", stored, new_root)
            graph.root_path = new_root
            return

    logger.warning("Stored root_path %s does not exist and could not be remapped", stored)


def list_modules(index_dir: str = ".code_index") -> str:
    """List all modules in the indexed project."""
    q = _get_query(index_dir)
    modules = q.list_modules()
    if not modules:
        return "No modules found in the index."
    lines = [f"# Modules ({len(modules)} files)\n"]
    for m in sorted(modules, key=lambda n: n.file_path):
        summary = f" — {m.summary}" if m.summary else ""
        lines.append(f"- `{m.file_path}`{summary}")
    return "\n".join(lines)


def get_file_structure(file_path: str, index_dir: str = ".code_index") -> str:
    """Show the structure of a specific file."""
    q = _get_query(index_dir)
    nodes = q.get_file_structure(file_path)
    if not nodes:
        return f"No nodes found for `{file_path}`. Check the path or re-index."
    lines = [f"# Structure of `{file_path}`\n"]
    for n in nodes:
        indent = "  " if n.node_type in (NodeType.METHOD,) else ""
        sig = f" — `{n.signature}`" if n.signature else ""
        summary = f"\n{indent}  {n.summary}" if n.summary else ""
        lines.append(f"{indent}- **{n.node_type.value}** `{n.name}` (L{n.line_start}-{n.line_end}){sig}{summary}")
    return "\n".join(lines)


def get_code(node_id: str, index_dir: str = ".code_index") -> str:
    """Get source code for a specific node."""
    q = _get_query(index_dir)
    node = q.get_node(node_id)
    if not node:
        return f"Node '{node_id}' not found."
    try:
        root = Path(q.graph.root_path)
        source = (root / node.file_path).read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        snippet = lines[node.line_start - 1 : node.line_end]
        header = f"# `{node.id}` ({node.node_type.value})\n"
        if node.summary:
            header += f"{node.summary}\n"
        header += f"\n```python\n"
        return header + "\n".join(snippet) + "\n```"
    except OSError:
        return f"Could not read source for `{node.id}`."


def get_element_context(node_id: str, index_dir: str = ".code_index") -> str:
    """Get a node with its parent, children, and relationships."""
    q = _get_query(index_dir)
    ctx = q.get_context(node_id)
    if "error" in ctx:
        return ctx["error"]

    node = ctx["node"]
    lines = [f"# Context for `{node.id}`\n"]
    lines.append(f"**Type:** {node.node_type.value}")
    lines.append(f"**File:** {node.file_path}:{node.line_start}")
    if node.signature:
        lines.append(f"**Signature:** `{node.signature}`")
    if node.summary:
        lines.append(f"**Summary:** {node.summary}")

    if ctx["parent"]:
        lines.append(f"\n**Parent:** `{ctx['parent'].id}`")

    if ctx["children"]:
        lines.append(f"\n**Children ({len(ctx['children'])}):**")
        for c in ctx["children"]:
            lines.append(f"  - `{c.name}` ({c.node_type.value})")

    out = [e for e in ctx["outgoing"] if e.edge_type != EdgeType.CONTAINS]
    if out:
        lines.append(f"\n**Dependencies ({len(out)}):**")
        for e in out:
            lines.append(f"  - {e.edge_type.value} → `{e.target}`")

    inc = [e for e in ctx["incoming"] if e.edge_type != EdgeType.CONTAINS]
    if inc:
        lines.append(f"\n**Used by ({len(inc)}):**")
        for e in inc:
            lines.append(f"  - {e.edge_type.value} ← `{e.source}`")

    return "\n".join(lines)


def search_entity(query: str, index_dir: str = ".code_index") -> str:
    """Search for classes, functions, and files by name, docstring, or summary."""
    q = _get_query(index_dir)
    results = q.search(query)
    if not results:
        return f"No results for '{query}'."
    lines = [f"# Search results for '{query}' ({len(results)} matches)\n"]
    for n in results[:20]:
        summary = f" — {n.summary}" if n.summary else ""
        lines.append(f"- `{n.id}` ({n.node_type.value}){summary}")
    if len(results) > 20:
        lines.append(f"\n... and {len(results) - 20} more")
    return "\n".join(lines)


def explore_structure(
    path: str | None = None, depth: int = 2, index_dir: str = ".code_index"
) -> str:
    """Explore project directory structure as a tree with summaries.

    If directory nodes are missing from the graph, they are built (and
    optionally summarised via LLM) on the fly, and the graph is saved back.
    """
    graph = load_graph(Path(index_dir))

    # Check whether directory nodes already exist
    has_dirs = any(n.node_type == NodeType.DIRECTORY for n in graph.nodes)
    needs_module_summaries = any(
        n.node_type == NodeType.MODULE and not n.summary for n in graph.nodes
    )
    if not has_dirs or needs_module_summaries:
        if not has_dirs:
            logger.info("No directory nodes found — building them now")
        if needs_module_summaries:
            logger.info("Module nodes missing summaries — summarizing now")
        _ensure_directory_nodes(graph, index_dir)

    q = GraphQuery(graph)
    tree = q.get_directory_tree(path, depth)
    if "error" in tree:
        return tree["error"]
    lines: list[str] = []
    _format_tree(tree, lines, indent=0)
    if not lines:
        return "No directory structure found. Re-index with directory support."
    return "\n".join(lines)


def _ensure_directory_nodes(graph, index_dir: str) -> None:
    """Build directory nodes (+ LLM summaries if available) and persist."""
    from coding_agent.code_indexer.graph.builder import GraphBuilder
    from coding_agent.code_indexer.models import IndexerConfig

    root = Path(graph.root_path)
    builder = GraphBuilder(root, config=IndexerConfig())

    # Build directory nodes only if they don't exist yet
    has_dirs = any(n.node_type == NodeType.DIRECTORY for n in graph.nodes)
    if not has_dirs:
        builder._build_directory_nodes(graph.nodes, graph.edges)

    # Attempt LLM summarisation of the new directory nodes
    try:
        from coding_agent.code_indexer.cli import _get_llm_service
        llm = _get_llm_service()
    except Exception:
        llm = None

    if llm:
        from coding_agent.code_indexer.summarizer import summarize_directories, summarize_modules
        summarize_modules(graph.nodes, graph.edges, llm)
        summarize_directories(graph.nodes, graph.edges, llm)

    # Rebuild internal indexes so queries see the new nodes/edges
    graph._build_indexes()

    # Persist so subsequent calls don't repeat this work
    try:
        save_graph(graph, Path(index_dir))
        logger.info("Saved graph with directory nodes to %s", index_dir)
    except Exception:
        logger.warning("Could not save updated graph", exc_info=True)


def _format_tree(node: dict, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    tag = "[D]" if node["type"] == "directory" else "[F]"
    summary = f" -- {node['summary']}" if node.get("summary") else ""
    lines.append(f"{prefix}{tag} {node['name']}{summary}")
    for child in node.get("children", []):
        _format_tree(child, lines, indent + 1)


def get_related_elements(node_id: str, index_dir: str = ".code_index") -> str:
    """Get all elements related to a node."""
    q = _get_query(index_dir)
    related = q.get_related(node_id)
    if not related:
        return f"No related elements for '{node_id}'."
    lines = [f"# Related to `{node_id}` ({len(related)} elements)\n"]
    for node, edge_type in related:
        summary = f" — {node.summary}" if node.summary else ""
        lines.append(f"- [{edge_type.value}] `{node.id}` ({node.node_type.value}){summary}")
    return "\n".join(lines)
