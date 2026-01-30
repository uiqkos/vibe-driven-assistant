"""Save/load ProjectGraph as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from coding_agent.code_indexer.models import ProjectGraph

DEFAULT_DIR = ".code_index"
DEFAULT_FILE = "graph.json"


def save_graph(graph: ProjectGraph, output_dir: Path | None = None) -> Path:
    """Save graph to JSON file. Returns the output path."""
    out = output_dir or Path(graph.root_path) / DEFAULT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / DEFAULT_FILE
    path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_graph(index_dir: Path) -> ProjectGraph:
    """Load graph from JSON file."""
    path = index_dir / DEFAULT_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProjectGraph.model_validate(data)
