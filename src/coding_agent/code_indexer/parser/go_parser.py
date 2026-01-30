"""Tree-sitter based parser for Go files."""

from __future__ import annotations

import logging
from pathlib import Path

from coding_agent.code_indexer.models import CodeNode, Edge, EdgeType, NodeType
from coding_agent.code_indexer.parser.base_parser import BaseParser

logger = logging.getLogger(__name__)


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_child_by_type(node, *types: str):
    for child in node.children:
        if child.type in types:
            return child
    return None


def _find_children_by_type(node, *types: str):
    return [c for c in node.children if c.type in types]


def _get_doc_comment(node, source_bytes: bytes) -> str:
    """Extract Go doc comment (consecutive // lines before a declaration)."""
    comments = []
    prev = node.prev_named_sibling
    while prev and prev.type == "comment":
        text = _node_text(prev, source_bytes).lstrip("/ ").rstrip()
        comments.insert(0, text)
        prev = prev.prev_named_sibling
    return "\n".join(comments)


class GoParser(BaseParser):
    """Parser for Go files using tree-sitter."""

    def __init__(self) -> None:
        self._parser = None

    def _get_parser(self):
        if self._parser is None:
            import tree_sitter_go as ts_go
            from tree_sitter import Language, Parser
            lang = Language(ts_go.language())
            self._parser = Parser(lang)
        return self._parser

    def get_supported_extensions(self) -> list[str]:
        return [".go"]

    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        rel = str(file_path.relative_to(root_path))
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return [], []

        source_bytes = source.encode("utf-8")
        parser = self._get_parser()
        tree = parser.parse(source_bytes)

        lines = source.splitlines()
        nodes: list[CodeNode] = []
        edges: list[Edge] = []

        mod_id = rel
        nodes.append(CodeNode(
            id=mod_id, name=file_path.stem, node_type=NodeType.MODULE,
            file_path=rel, line_start=1, line_end=len(lines),
        ))

        for child in tree.root_node.children:
            if child.type == "function_declaration":
                self._handle_function(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "method_declaration":
                self._handle_method(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "type_declaration":
                self._handle_type_decl(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "import_declaration":
                self._handle_import(child, mod_id, source_bytes, edges)

        return nodes, edges

    def _handle_function(self, node, mod_id: str, rel: str, source_bytes: bytes,
                         nodes: list[CodeNode], edges: list[Edge]) -> None:
        name_node = _find_child_by_type(node, "identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)
        func_id = f"{rel}::{name}"
        doc = _get_doc_comment(node, source_bytes)

        params = _find_child_by_type(node, "parameter_list")
        result = _find_child_by_type(node, "result")
        sig = f"func {name}({_node_text(params, source_bytes) if params else ''})"
        if result:
            sig += f" {_node_text(result, source_bytes)}"

        nodes.append(CodeNode(
            id=func_id, name=name, node_type=NodeType.FUNCTION, file_path=rel,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=sig, docstring=doc, parent_id=mod_id,
        ))
        edges.append(Edge(source=mod_id, target=func_id, edge_type=EdgeType.CONTAINS))
        self._extract_calls(node, func_id, source_bytes, edges)

    def _handle_method(self, node, mod_id: str, rel: str, source_bytes: bytes,
                       nodes: list[CodeNode], edges: list[Edge]) -> None:
        name_node = _find_child_by_type(node, "field_identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)

        # Get receiver type
        receiver = _find_child_by_type(node, "parameter_list")
        receiver_type = ""
        if receiver:
            # The receiver param list contains (name Type) or (*Type)
            for param in receiver.children:
                if param.type == "parameter_declaration":
                    type_node = _find_child_by_type(param, "type_identifier", "pointer_type")
                    if type_node:
                        receiver_type = _node_text(type_node, source_bytes).lstrip("*")
                        break

        struct_name = receiver_type or "<unknown>"
        meth_id = f"{rel}::{struct_name}.{name}"
        doc = _get_doc_comment(node, source_bytes)

        # params is the second parameter_list (after receiver)
        param_lists = _find_children_by_type(node, "parameter_list")
        params = param_lists[1] if len(param_lists) > 1 else None
        result = _find_child_by_type(node, "result")
        sig = f"func ({struct_name}) {name}({_node_text(params, source_bytes) if params else ''})"
        if result:
            sig += f" {_node_text(result, source_bytes)}"

        nodes.append(CodeNode(
            id=meth_id, name=f"{struct_name}.{name}", node_type=NodeType.METHOD,
            file_path=rel, line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=sig, docstring=doc, parent_id=mod_id,
        ))
        edges.append(Edge(source=mod_id, target=meth_id, edge_type=EdgeType.CONTAINS))
        self._extract_calls(node, meth_id, source_bytes, edges)

    def _handle_type_decl(self, node, mod_id: str, rel: str, source_bytes: bytes,
                          nodes: list[CodeNode], edges: list[Edge]) -> None:
        for spec in _find_children_by_type(node, "type_spec"):
            name_node = _find_child_by_type(spec, "type_identifier")
            if not name_node:
                continue
            name = _node_text(name_node, source_bytes)
            doc = _get_doc_comment(node, source_bytes)

            # Determine if struct or interface
            struct_type = _find_child_by_type(spec, "struct_type")
            iface_type = _find_child_by_type(spec, "interface_type")

            if struct_type:
                node_type = NodeType.STRUCT
                type_id = f"{rel}::{name}"
                nodes.append(CodeNode(
                    id=type_id, name=name, node_type=node_type, file_path=rel,
                    line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                    docstring=doc, parent_id=mod_id,
                ))
                edges.append(Edge(source=mod_id, target=type_id, edge_type=EdgeType.CONTAINS))

                # Embedded structs → INHERITS
                field_list = _find_child_by_type(struct_type, "field_declaration_list")
                if field_list:
                    for field in _find_children_by_type(field_list, "field_declaration"):
                        # Embedded field has no name, just a type
                        children = [c for c in field.children if c.type != "comment"]
                        if len(children) == 1 and children[0].type in ("type_identifier", "qualified_type"):
                            embedded = _node_text(children[0], source_bytes)
                            edges.append(Edge(source=type_id, target=embedded, edge_type=EdgeType.INHERITS))

            elif iface_type:
                node_type = NodeType.INTERFACE
                type_id = f"{rel}::{name}"
                nodes.append(CodeNode(
                    id=type_id, name=name, node_type=node_type, file_path=rel,
                    line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                    docstring=doc, parent_id=mod_id,
                ))
                edges.append(Edge(source=mod_id, target=type_id, edge_type=EdgeType.CONTAINS))

                # Embedded interfaces: look for type_elem children containing type_identifier
                for child in iface_type.children:
                    if child.type == "type_elem":
                        tid = _find_child_by_type(child, "type_identifier")
                        if tid:
                            edges.append(Edge(source=type_id, target=_node_text(tid, source_bytes),
                                              edge_type=EdgeType.INHERITS))
                    elif child.type == "type_identifier":
                        edges.append(Edge(source=type_id, target=_node_text(child, source_bytes),
                                          edge_type=EdgeType.INHERITS))

    def _handle_import(self, node, mod_id: str, source_bytes: bytes, edges: list[Edge]) -> None:
        # Single import or import block
        for spec in node.children:
            if spec.type == "import_spec":
                path_node = _find_child_by_type(spec, "interpreted_string_literal")
                if path_node:
                    target = _node_text(path_node, source_bytes).strip('"')
                    edges.append(Edge(source=mod_id, target=target, edge_type=EdgeType.IMPORTS))
            elif spec.type == "import_spec_list":
                for imp in _find_children_by_type(spec, "import_spec"):
                    path_node = _find_child_by_type(imp, "interpreted_string_literal")
                    if path_node:
                        target = _node_text(path_node, source_bytes).strip('"')
                        edges.append(Edge(source=mod_id, target=target, edge_type=EdgeType.IMPORTS))

    def _extract_calls(self, node, caller_id: str, source_bytes: bytes, edges: list[Edge]) -> None:
        if node is None:
            return
        cursor = node.walk()
        reached_root = False
        while not reached_root:
            current = cursor.node
            if current.type == "call_expression":
                func_node = _find_child_by_type(current, "identifier", "selector_expression", "field_identifier")
                if func_node:
                    name = _node_text(func_node, source_bytes)
                    if name:
                        edges.append(Edge(source=caller_id, target=name, edge_type=EdgeType.CALLS))

            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue
            while True:
                if not cursor.goto_parent():
                    reached_root = True
                    break
                if cursor.node == node:
                    reached_root = True
                    break
                if cursor.goto_next_sibling():
                    break
