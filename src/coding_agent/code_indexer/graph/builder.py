"""Orchestrator: scan files → parse → resolve → summarize → build graph."""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from coding_agent.code_indexer.models import (
    CodeNode,
    Edge,
    EdgeType,
    FileChecksum,
    IndexerConfig,
    NodeType,
    ProjectGraph,
)
from coding_agent.code_indexer.parser.ast_parser import PythonAstParser
from coding_agent.code_indexer.parser.llm_parser import parse_file_with_llm
from coding_agent.code_indexer.parser.registry import ParserRegistry
from coding_agent.code_indexer.parser.resolver import resolve_edges
from coding_agent.code_indexer.summarizer import summarize_directories, summarize_modules, summarize_nodes


def _build_default_registry() -> ParserRegistry:
    """Create a registry with all available parsers."""
    registry = ParserRegistry()
    registry.register(PythonAstParser())
    try:
        from coding_agent.code_indexer.parser.javascript_parser import JavaScriptParser
        registry.register(JavaScriptParser())
    except ImportError:
        logger.debug("tree-sitter JS/TS not available, skipping JavaScript parser")
    try:
        from coding_agent.code_indexer.parser.go_parser import GoParser
        registry.register(GoParser())
    except ImportError:
        logger.debug("tree-sitter Go not available, skipping Go parser")
    try:
        from coding_agent.code_indexer.parser.rust_parser import RustParser
        registry.register(RustParser())
    except ImportError:
        logger.debug("tree-sitter Rust not available, skipping Rust parser")
    return registry


class GraphBuilder:
    """Builds a ProjectGraph from a project directory."""

    def __init__(
        self,
        root_path: Path,
        config: IndexerConfig | None = None,
        llm_service: object | None = None,
        registry: ParserRegistry | None = None,
    ) -> None:
        self.root_path = root_path.resolve()
        self.config = config or IndexerConfig()
        self.llm_service = llm_service
        self.registry = registry or _build_default_registry()

    def build(self) -> ProjectGraph:
        """Full build: scan → parse → resolve → summarize."""
        files = self._scan_files()
        logger.info("Scanned %d files", len(files))
        all_nodes: list[CodeNode] = []
        all_edges: list[Edge] = []
        checksums: list[FileChecksum] = []

        for i, f in enumerate(files, 1):
            rel = f.relative_to(self.root_path)
            logger.info("[%d/%d] Parsing %s", i, len(files), rel)
            nodes, edges = self._parse_single_file(f)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            checksums.append(FileChecksum.compute(f, self.root_path))

        logger.info("Parsed %d nodes, %d edges", len(all_nodes), len(all_edges))
        all_edges = resolve_edges(all_nodes, all_edges)
        logger.info("Resolved edges: %d total", len(all_edges))

        self._build_directory_nodes(all_nodes, all_edges)

        if self.config.use_llm_summaries and self.llm_service:
            summarize_nodes(
                all_nodes,
                self.root_path,
                self.llm_service,
                batch_size=self.config.batch_size,
            )
            summarize_modules(all_nodes, all_edges, self.llm_service)
            summarize_directories(all_nodes, all_edges, self.llm_service)
        else:
            logger.info("Skipping summarization (llm_summaries=%s, llm_service=%s)",
                        self.config.use_llm_summaries, self.llm_service is not None)

        return ProjectGraph(
            root_path=str(self.root_path),
            nodes=all_nodes,
            edges=all_edges,
            checksums=checksums,
        )

    def update(self, existing: ProjectGraph) -> ProjectGraph:
        """Incremental update: only re-parse changed files."""
        files = self._scan_files()
        logger.info("Incremental update: scanning %d files for changes", len(files))
        changed: list[Path] = []
        unchanged_files: set[str] = set()

        for f in files:
            rel = str(f.relative_to(self.root_path))
            current_md5 = FileChecksum.compute(f, self.root_path).md5
            stored_md5 = existing.get_checksum(rel)
            if stored_md5 != current_md5:
                changed.append(f)
            else:
                unchanged_files.add(rel)

        # Keep nodes/edges from unchanged files
        kept_nodes = [n for n in existing.nodes if n.file_path in unchanged_files]
        kept_edges = [
            e for e in existing.edges
            if e.source.split("::")[0] in unchanged_files
        ]
        kept_checksums = [c for c in existing.checksums if c.file_path in unchanged_files]

        logger.info("Found %d changed files, %d unchanged", len(changed), len(unchanged_files))
        for f in changed:
            logger.info("  Changed: %s", f.relative_to(self.root_path))

        # Parse changed files
        new_nodes: list[CodeNode] = []
        new_edges: list[Edge] = []
        new_checksums: list[FileChecksum] = []

        for i, f in enumerate(changed, 1):
            rel = f.relative_to(self.root_path)
            logger.info("[%d/%d] Re-parsing %s", i, len(changed), rel)
            nodes, edges = self._parse_single_file(f)
            new_nodes.extend(nodes)
            new_edges.extend(edges)
            new_checksums.append(FileChecksum.compute(f, self.root_path))

        all_nodes = kept_nodes + new_nodes
        all_edges = kept_edges + new_edges
        all_edges = resolve_edges(all_nodes, all_edges)

        # Rebuild directory nodes from scratch (remove old ones first)
        all_nodes = [n for n in all_nodes if n.node_type != NodeType.DIRECTORY]
        all_edges = [e for e in all_edges if not e.source.startswith("dir:")]
        self._build_directory_nodes(all_nodes, all_edges)

        if self.config.use_llm_summaries and self.llm_service:
            # Summarize all nodes missing a summary (new + previously unsummarized)
            unsummarized = [n for n in all_nodes if not n.summary]
            logger.info(
                "Nodes to summarize: %d (new: %d, previously unsummarized: %d)",
                len(unsummarized), len(new_nodes), len(unsummarized) - len(new_nodes),
            )
            summarize_nodes(
                all_nodes,
                self.root_path,
                self.llm_service,
                batch_size=self.config.batch_size,
            )
            summarize_modules(all_nodes, all_edges, self.llm_service)
            summarize_directories(all_nodes, all_edges, self.llm_service)

        return ProjectGraph(
            root_path=str(self.root_path),
            nodes=all_nodes,
            edges=all_edges,
            checksums=kept_checksums + new_checksums,
        )

    def _parse_single_file(self, f: Path) -> tuple[list[CodeNode], list[Edge]]:
        """Parse a file using the appropriate parser from the registry."""
        parser = self.registry.get_parser(f)
        if parser:
            nodes, edges = parser.parse_file(f, self.root_path)
        else:
            nodes, edges = [], []

        # LLM fallback for Python files only
        if not nodes and f.suffix == ".py" and self.config.use_llm_fallback_parser and self.llm_service:
            rel = f.relative_to(self.root_path)
            logger.info("AST failed, LLM fallback: %s", rel)
            nodes, edges = parse_file_with_llm(f, self.root_path, self.llm_service)

        return nodes, edges

    def _build_directory_nodes(
        self, nodes: list[CodeNode], edges: list[Edge]
    ) -> None:
        """Create DIRECTORY nodes and CONTAINS edges for the directory hierarchy.

        Mutates ``nodes`` and ``edges`` in place.
        """
        # Collect all unique directory paths from MODULE nodes
        dir_paths: set[str] = set()
        module_dir: dict[str, str] = {}  # module file_path -> its directory
        for n in nodes:
            if n.node_type == NodeType.MODULE:
                parts = n.file_path.split("/")
                if len(parts) > 1:
                    dir_path = "/".join(parts[:-1])
                    module_dir[n.file_path] = dir_path
                else:
                    module_dir[n.file_path] = ""
                # Add all ancestor dirs
                for i in range(1, len(parts)):
                    dir_paths.add("/".join(parts[:i]))

        # Also add root "" if there are any top-level modules
        if any(v == "" for v in module_dir.values()):
            dir_paths.add("")

        # Create DIRECTORY nodes
        dir_nodes: dict[str, CodeNode] = {}
        for dp in sorted(dir_paths):
            name = dp.split("/")[-1] if dp else "."
            node_id = f"dir:{dp}" if dp else "dir:."
            parent_dp = "/".join(dp.split("/")[:-1]) if "/" in dp else ("" if dp else None)
            parent_id: str | None = None
            if parent_dp is not None:
                parent_id = f"dir:{parent_dp}" if parent_dp else "dir:."
            dn = CodeNode(
                id=node_id,
                name=name,
                node_type=NodeType.DIRECTORY,
                file_path=dp if dp else ".",
                line_start=0,
                line_end=0,
                parent_id=parent_id,
            )
            dir_nodes[dp] = dn

        nodes.extend(dir_nodes.values())

        # Re-parent MODULE nodes to their directory and add CONTAINS edges
        for n in nodes:
            if n.node_type == NodeType.MODULE and n.file_path in module_dir:
                dp = module_dir[n.file_path]
                if dp in dir_nodes:
                    n.parent_id = dir_nodes[dp].id
                    edges.append(Edge(
                        source=dir_nodes[dp].id,
                        target=n.id,
                        edge_type=EdgeType.CONTAINS,
                    ))

        # Add CONTAINS edges for directory→subdirectory
        for _dp, dn in dir_nodes.items():
            if dn.parent_id and dn.parent_id.startswith("dir:"):
                edges.append(Edge(
                    source=dn.parent_id,
                    target=dn.id,
                    edge_type=EdgeType.CONTAINS,
                ))

        logger.info("Built %d directory nodes", len(dir_nodes))

    def _scan_files(self) -> list[Path]:
        """Find all files with registered extensions, excluding configured patterns."""
        extensions = self.registry.get_all_extensions()
        files = []
        for ext in extensions:
            pattern = f"*{ext}"
            for f in self.root_path.rglob(pattern):
                rel = str(f.relative_to(self.root_path))
                if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.name, pat)
                       or any(fnmatch.fnmatch(part, pat) for part in f.parts)
                       for pat in self.config.exclude_patterns):
                    continue
                files.append(f)
        return sorted(set(files))
