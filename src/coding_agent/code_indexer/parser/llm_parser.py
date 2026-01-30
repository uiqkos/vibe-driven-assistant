"""LLM fallback parser for files that fail AST parsing."""

from __future__ import annotations

import json
from pathlib import Path

from coding_agent.code_indexer.models import CodeNode, Edge, EdgeType, NodeType

SYSTEM_PROMPT = """\
You are a code analysis assistant. Given a Python source file, extract its structure as JSON.
Return ONLY valid JSON with this schema:
{
  "nodes": [
    {"name": "...", "type": "module|class|function|method", "line_start": N, "line_end": N,
     "signature": "...", "docstring": "...", "parent": null|"ClassName"}
  ],
  "edges": [
    {"source": "...", "target": "...", "type": "imports|calls|inherits|contains"}
  ]
}
"""


def parse_file_with_llm(
    file_path: Path,
    root_path: Path,
    llm_service: object,  # LLMService
) -> tuple[list[CodeNode], list[Edge]]:
    """Use LLM to extract structure from a file that failed AST parsing."""
    rel = str(file_path.relative_to(root_path))
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], []

    prompt = f"Analyze this Python file and extract its structure:\n\n```python\n{source[:8000]}\n```"

    try:
        raw = llm_service.generate(prompt, system_prompt=SYSTEM_PROMPT)  # type: ignore[attr-defined]
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return [], []

    nodes: list[CodeNode] = []
    edges: list[Edge] = []

    type_map = {
        "module": NodeType.MODULE,
        "class": NodeType.CLASS,
        "function": NodeType.FUNCTION,
        "method": NodeType.METHOD,
    }

    for n in data.get("nodes", []):
        ntype = type_map.get(n.get("type", ""), NodeType.FUNCTION)
        parent = n.get("parent")
        if ntype == NodeType.MODULE:
            nid = rel
        elif ntype == NodeType.METHOD and parent:
            nid = f"{rel}::{parent}.{n['name']}"
        else:
            nid = f"{rel}::{n['name']}"
        parent_id = f"{rel}::{parent}" if parent else (rel if ntype != NodeType.MODULE else None)
        nodes.append(
            CodeNode(
                id=nid,
                name=n.get("name", ""),
                node_type=ntype,
                file_path=rel,
                line_start=n.get("line_start", 1),
                line_end=n.get("line_end", 1),
                signature=n.get("signature", ""),
                docstring=n.get("docstring", ""),
                parent_id=parent_id,
            )
        )

    edge_type_map = {
        "imports": EdgeType.IMPORTS,
        "calls": EdgeType.CALLS,
        "inherits": EdgeType.INHERITS,
        "contains": EdgeType.CONTAINS,
    }
    for e in data.get("edges", []):
        etype = edge_type_map.get(e.get("type", ""))
        if etype:
            edges.append(Edge(source=e["source"], target=e["target"], edge_type=etype))

    return nodes, edges
