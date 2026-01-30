"""RAG search tools for the agent tool registry."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from coding_agent.rag.config import RAGConfig
from coding_agent.rag.stores import CodeStore, SearchResult, SummaryStore
from coding_agent.tools import Tool


def _format_code_result(r: SearchResult) -> str:
    fp = r.metadata.get("file_path", "?")
    start = r.metadata.get("start_line", "?")
    end = r.metadata.get("end_line", "?")
    lang = r.metadata.get("language", "")
    header = f"[Score: {r.score:.2f}] {fp}:{start}-{end} ({lang})"
    separator = "\u2500" * min(len(header), 50)
    content = r.content
    if len(content) > 2000:
        content = content[:2000] + "\n... (truncated)"
    return f"{header}\n{separator}\n{content}"


def _format_summary_result(r: SearchResult) -> str:
    node_id = r.metadata.get("node_id", r.id)
    fp = r.metadata.get("file_path", "?")
    nt = r.metadata.get("node_type", "?")
    header = f"[Score: {r.score:.2f}] {node_id} ({nt}) in {fp}"
    return f"{header}\n  {r.content}"


def _search_code(index_dir: str, query: str, top_k: int = 10, file_filter: str = "") -> str:
    config = RAGConfig()
    store = CodeStore(Path(index_dir), config)
    where = None
    results = store.query(query, top_k=top_k, where=where)
    if file_filter:
        results = [r for r in results if fnmatch.fnmatch(r.metadata.get("file_path", ""), file_filter)]
    if not results:
        return "No code results found."
    parts = [_format_code_result(r) for r in results]
    return f"Found {len(results)} code chunks:\n\n" + "\n\n".join(parts)


def _search_semantic(index_dir: str, query: str, top_k: int = 10) -> str:
    config = RAGConfig()
    store = SummaryStore(Path(index_dir), config)
    results = store.query(query, top_k=top_k)
    if not results:
        return "No summary results found."
    parts = [_format_summary_result(r) for r in results]
    return f"Found {len(results)} summaries:\n\n" + "\n".join(parts)


def _search_hybrid(index_dir: str, query: str, top_k: int = 10) -> str:
    config = RAGConfig()
    code_store = CodeStore(Path(index_dir), config)
    summary_store = SummaryStore(Path(index_dir), config)

    code_results = code_store.query(query, top_k=top_k)
    summary_results = summary_store.query(query, top_k=top_k)

    # Deduplicate by file_path, preferring higher scores
    seen_files: set[str] = set()
    parts: list[tuple[float, str]] = []

    for r in code_results:
        fp = r.metadata.get("file_path", "")
        parts.append((r.score, _format_code_result(r)))
        seen_files.add(fp)

    for r in summary_results:
        fp = r.metadata.get("file_path", "")
        if fp not in seen_files:
            parts.append((r.score, _format_summary_result(r)))
            seen_files.add(fp)
        else:
            # Still include summary even for seen files (different info)
            parts.append((r.score, _format_summary_result(r)))

    parts.sort(key=lambda x: x[0], reverse=True)
    parts = parts[:top_k]

    if not parts:
        return "No results found."
    formatted = [p[1] for p in parts]
    return f"Found {len(formatted)} hybrid results:\n\n" + "\n\n".join(formatted)


def create_rag_tools(index_dir: str) -> list[Tool]:
    """Create RAG search tools for the agent ToolRegistry."""
    return [
        Tool(
            name="rag_search_code",
            description=(
                "Semantic code search via vector embeddings. "
                "Finds code by MEANING, not just name — understands natural language. "
                "Returns code snippets with exact file paths and line numbers. "
                "Works for all file types: Python, JS/TS, Go, Rust, config files, docs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language description of the code you're looking for. "
                            "Be specific about the behavior or concept, not just a class name. "
                            "Good: 'how entity position is updated during movement'. "
                            "Bad: 'position' (too vague)."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 10). Use 20+ for broad searches.",
                        "default": 10,
                    },
                    "file_filter": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py', 'components/*.py'). Optional.",
                        "default": "",
                    },
                },
                "required": ["query"],
            },
            execute=lambda args: _search_code(
                index_dir,
                args["query"],
                args.get("top_k", 10),
                args.get("file_filter", ""),
            ),
        ),
        Tool(
            name="rag_search_semantic",
            description=(
                "Search AI-generated summaries of modules, classes, and functions by meaning. "
                "Finds components by PURPOSE and BEHAVIOR without reading source code. "
                "Returns node IDs (usable with get_node_info/get_source_code), file paths, and types. "
                "Use over rag_search_code when you need to understand WHAT code does, not HOW."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language question about functionality, responsibility, or architecture. "
                            "Good: 'which component manages entity health and damage'. "
                            "Bad: 'hp' (too short, use search_entity for name lookups)."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            execute=lambda args: _search_semantic(
                index_dir,
                args["query"],
                args.get("top_k", 10),
            ),
        ),
        Tool(
            name="rag_search_hybrid",
            description=(
                "Combined search across BOTH code snippets AND summaries in one call. "
                "Returns a ranked mix of code chunks (with line numbers) and component descriptions. "
                "Best first tool when investigating a bug or feature — gives both code and context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language query about a bug, feature, or concept. "
                            "Be descriptive: 'entity position not updating after movement action' "
                            "works better than just 'position'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 10). Use 15-20 for complex investigations.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            execute=lambda args: _search_hybrid(
                index_dir,
                args["query"],
                args.get("top_k", 10),
            ),
        ),
    ]
