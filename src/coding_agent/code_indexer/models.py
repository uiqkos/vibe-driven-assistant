"""Data models for the code indexer."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class NodeType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    DIRECTORY = "directory"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    TRAIT = "trait"


class EdgeType(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    CONTAINS = "contains"  # module->class, class->method
    IMPLEMENTS = "implements"


class CodeNode(BaseModel):
    id: str  # e.g. "src/foo.py::MyClass.method"
    name: str
    node_type: NodeType
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    summary: str = ""
    parent_id: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.node_type.value}) in {self.file_path}"


class Edge(BaseModel):
    source: str  # node id
    target: str  # node id
    edge_type: EdgeType


class FileChecksum(BaseModel):
    file_path: str
    md5: str

    @staticmethod
    def compute(path: Path, root: Path | None = None) -> FileChecksum:
        content = path.read_bytes()
        rel = str(path.relative_to(root)) if root else str(path)
        return FileChecksum(
            file_path=rel,
            md5=hashlib.md5(content).hexdigest(),
        )


class ProjectGraph(BaseModel):
    root_path: str
    nodes: list[CodeNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    checksums: list[FileChecksum] = Field(default_factory=list)
    git_head: str = ""  # commit SHA at index time

    # Indexes — rebuilt on load, not serialized
    _node_index: dict[str, CodeNode] = {}
    _file_index: dict[str, list[CodeNode]] = {}
    _adj: dict[str, list[Edge]] = {}
    _rev_adj: dict[str, list[Edge]] = {}

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _build_indexes(self) -> "ProjectGraph":
        self._node_index = {n.id: n for n in self.nodes}
        self._file_index = {}
        for n in self.nodes:
            self._file_index.setdefault(n.file_path, []).append(n)
        self._adj = {}
        self._rev_adj = {}
        for e in self.edges:
            self._adj.setdefault(e.source, []).append(e)
            self._rev_adj.setdefault(e.target, []).append(e)
        return self

    def get_node(self, node_id: str) -> CodeNode | None:
        # Exact match
        if node_id in self._node_index:
            return self._node_index[node_id]
        # Suffix match: "provider.py::MyClass" matches "src/flask/json/provider.py::MyClass"
        normalized = node_id.replace("\\", "/").lstrip("./")
        for stored_id, node in self._node_index.items():
            if stored_id.endswith(normalized) or normalized.endswith(stored_id):
                return node
        return None

    def get_file_nodes(self, file_path: str) -> list[CodeNode]:
        # Exact match first
        if file_path in self._file_index:
            return self._file_index[file_path]
        # Suffix match: "provider.py" matches "src/flask/json/provider.py"
        normalized = file_path.replace("\\", "/").lstrip("./")
        for stored_path, nodes in self._file_index.items():
            if stored_path.endswith(normalized) or normalized.endswith(stored_path):
                return nodes
        # Basename match: "provider.py" matches any file named provider.py
        basename = normalized.rsplit("/", 1)[-1]
        for stored_path, nodes in self._file_index.items():
            if stored_path.rsplit("/", 1)[-1] == basename:
                return nodes
        return []

    def get_outgoing(self, node_id: str) -> list[Edge]:
        return self._adj.get(node_id, [])

    def get_incoming(self, node_id: str) -> list[Edge]:
        return self._rev_adj.get(node_id, [])

    def get_checksum(self, file_path: str) -> str | None:
        for c in self.checksums:
            if c.file_path == file_path:
                return c.md5
        return None


class IndexerConfig(BaseModel):
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".tox",
            "*.egg-info",
        ]
    )
    batch_size: int = 10
    use_llm_summaries: bool = True
    use_llm_fallback_parser: bool = True
    enabled_languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "go", "rust"]
    )
