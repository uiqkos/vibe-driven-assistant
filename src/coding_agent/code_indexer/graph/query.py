"""Query interface for the project graph."""

from __future__ import annotations

from coding_agent.code_indexer.models import CodeNode, EdgeType, NodeType, ProjectGraph


class GraphQuery:
    """Query methods over a ProjectGraph."""

    def __init__(self, graph: ProjectGraph) -> None:
        self.graph = graph

    def list_modules(self) -> list[CodeNode]:
        """Return all module-level nodes."""
        return [n for n in self.graph.nodes if n.node_type == NodeType.MODULE]

    def get_file_structure(self, file_path: str) -> list[CodeNode]:
        """Return all nodes in a file, sorted by line number."""
        nodes = self.graph.get_file_nodes(file_path)
        return sorted(nodes, key=lambda n: n.line_start)

    def get_node(self, node_id: str) -> CodeNode | None:
        return self.graph.get_node(node_id)

    def search(self, query: str) -> list[CodeNode]:
        """Search nodes by substring match on name, docstring, and summary."""
        q = query.lower()
        results = []
        for n in self.graph.nodes:
            text = f"{n.name} {n.docstring} {n.summary}".lower()
            if q in text:
                results.append(n)
        return results

    def get_context(self, node_id: str) -> dict:
        """Get a node with its parent, children, and related edges."""
        node = self.graph.get_node(node_id)
        if not node:
            return {"error": f"Node '{node_id}' not found"}

        parent = self.graph.get_node(node.parent_id) if node.parent_id else None
        outgoing = self.graph.get_outgoing(node_id)
        incoming = self.graph.get_incoming(node_id)
        children = [
            e.target for e in outgoing if e.edge_type == EdgeType.CONTAINS
        ]
        child_nodes = [self.graph.get_node(c) for c in children]

        return {
            "node": node,
            "parent": parent,
            "children": [c for c in child_nodes if c],
            "outgoing": outgoing,
            "incoming": incoming,
        }

    def get_related(self, node_id: str) -> list[tuple[CodeNode, EdgeType]]:
        """Get all nodes related to this one via any edge."""
        related: list[tuple[CodeNode, EdgeType]] = []
        for e in self.graph.get_outgoing(node_id):
            n = self.graph.get_node(e.target)
            if n:
                related.append((n, e.edge_type))
        for e in self.graph.get_incoming(node_id):
            n = self.graph.get_node(e.source)
            if n:
                related.append((n, e.edge_type))
        return related

    def get_dependencies(self, node_id: str) -> list[CodeNode]:
        """Get nodes that this node depends on (imports/calls/inherits)."""
        deps = []
        for e in self.graph.get_outgoing(node_id):
            if e.edge_type in (EdgeType.IMPORTS, EdgeType.CALLS, EdgeType.INHERITS):
                n = self.graph.get_node(e.target)
                if n:
                    deps.append(n)
        return deps

    def get_dependents(self, node_id: str) -> list[CodeNode]:
        """Get nodes that depend on this node."""
        deps = []
        for e in self.graph.get_incoming(node_id):
            if e.edge_type in (EdgeType.IMPORTS, EdgeType.CALLS, EdgeType.INHERITS):
                n = self.graph.get_node(e.source)
                if n:
                    deps.append(n)
        return deps

    def get_directory_tree(self, path: str | None = None, depth: int = 2) -> dict:
        """Build a nested dict representing the directory tree.

        Args:
            path: Directory path to start from (None for root).
            depth: Maximum recursion depth.

        Returns:
            ``{name, type, summary, children: [...]}``
        """
        if path is None:
            # Find root directory node (no parent_id or parent is None)
            root = None
            for n in self.graph.nodes:
                if n.node_type == NodeType.DIRECTORY and n.parent_id is None:
                    root = n
                    break
            if root is None:
                return {"name": ".", "type": "directory", "summary": "", "children": []}
            node_id = root.id
        else:
            node_id = f"dir:{path}"
            root = self.graph.get_node(node_id)
            if root is None:
                return {"error": f"Directory '{path}' not found"}

        return self._build_tree(root, depth)

    def _build_tree(self, node: CodeNode, depth: int) -> dict:
        result: dict = {
            "name": node.name,
            "type": "directory" if node.node_type == NodeType.DIRECTORY else "file",
            "summary": node.summary or "",
        }
        if depth <= 0 or node.node_type != NodeType.DIRECTORY:
            if node.node_type == NodeType.DIRECTORY:
                result["children"] = []
            return result

        children: list[dict] = []
        for e in self.graph.get_outgoing(node.id):
            if e.edge_type == EdgeType.CONTAINS:
                child = self.graph.get_node(e.target)
                if child:
                    children.append(self._build_tree(child, depth - 1))

        # Sort: directories first, then files, alphabetically
        children.sort(key=lambda c: (0 if c["type"] == "directory" else 1, c["name"]))
        result["children"] = children
        return result
