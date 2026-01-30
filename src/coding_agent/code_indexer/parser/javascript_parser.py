"""Tree-sitter based parser for JavaScript and TypeScript files."""

from __future__ import annotations

import logging
from pathlib import Path

from coding_agent.code_indexer.models import CodeNode, Edge, EdgeType, NodeType
from coding_agent.code_indexer.parser.base_parser import BaseParser

logger = logging.getLogger(__name__)

# Node type names in tree-sitter grammars
_CLASS_TYPES = {"class_declaration", "class"}
_FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
}
_METHOD_TYPES = {"method_definition"}
_INTERFACE_TYPES = {"interface_declaration"}
_ENUM_TYPES = {"enum_declaration"}
_IMPORT_TYPES = {"import_statement"}


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_child_by_type(node, *types: str):
    for child in node.children:
        if child.type in types:
            return child
    return None


def _find_children_by_type(node, *types: str):
    return [c for c in node.children if c.type in types]


def _get_jsdoc(node, source_bytes: bytes) -> str:
    """Extract JSDoc comment preceding a node."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = _node_text(prev, source_bytes)
        if text.startswith("/**"):
            # Strip /** ... */ and leading * on each line
            lines = text.split("\n")
            cleaned = []
            for line in lines:
                line = line.strip()
                if line in ("/**", "*/"):
                    continue
                line = line.lstrip("* ").rstrip()
                if line:
                    cleaned.append(line)
            return "\n".join(cleaned)
    return ""


class JavaScriptParser(BaseParser):
    """Parser for JS/TS files using tree-sitter."""

    def __init__(self) -> None:
        self._js_parser = None
        self._ts_parser = None
        self._tsx_parser = None

    def _get_parser_and_language(self, suffix: str):
        """Lazily initialize tree-sitter parsers."""
        from tree_sitter import Language, Parser

        if suffix in (".ts", ".tsx"):
            if suffix == ".tsx":
                if self._tsx_parser is None:
                    import tree_sitter_typescript as ts_ts
                    lang = Language(ts_ts.language_tsx())
                    p = Parser(lang)
                    self._tsx_parser = p
                return self._tsx_parser
            else:
                if self._ts_parser is None:
                    import tree_sitter_typescript as ts_ts
                    lang = Language(ts_ts.language_typescript())
                    p = Parser(lang)
                    self._ts_parser = p
                return self._ts_parser
        else:
            if self._js_parser is None:
                import tree_sitter_javascript as ts_js
                lang = Language(ts_js.language())
                p = Parser(lang)
                self._js_parser = p
            return self._js_parser

    def get_supported_extensions(self) -> list[str]:
        return [".js", ".jsx", ".ts", ".tsx"]

    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        rel = str(file_path.relative_to(root_path))
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return [], []

        source_bytes = source.encode("utf-8")
        parser = self._get_parser_and_language(file_path.suffix)
        tree = parser.parse(source_bytes)

        lines = source.splitlines()
        nodes: list[CodeNode] = []
        edges: list[Edge] = []

        # Module node
        mod_id = rel
        nodes.append(
            CodeNode(
                id=mod_id,
                name=file_path.stem,
                node_type=NodeType.MODULE,
                file_path=rel,
                line_start=1,
                line_end=len(lines),
            )
        )

        self._walk_top_level(tree.root_node, mod_id, rel, source_bytes, nodes, edges)
        return nodes, edges

    def _walk_top_level(
        self,
        root,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        for child in root.children:
            # Handle export wrappers
            actual = child
            if child.type in ("export_statement", "export_default_declaration"):
                # The actual declaration is a child of the export
                decl = _find_child_by_type(
                    child,
                    "class_declaration", "class", "function_declaration",
                    "generator_function_declaration", "interface_declaration",
                    "enum_declaration", "lexical_declaration",
                )
                if decl:
                    actual = decl
                else:
                    # Could be `export default function() {}` or arrow
                    actual = child

            if actual.type in _CLASS_TYPES:
                self._handle_class(actual, mod_id, rel, source_bytes, nodes, edges)
            elif actual.type in _FUNCTION_TYPES:
                self._handle_function(actual, mod_id, rel, source_bytes, nodes, edges)
            elif actual.type in _INTERFACE_TYPES:
                self._handle_interface(actual, mod_id, rel, source_bytes, nodes, edges)
            elif actual.type in _ENUM_TYPES:
                self._handle_enum(actual, mod_id, rel, source_bytes, nodes, edges)
            elif actual.type == "lexical_declaration":
                # const Foo = () => {} or const Foo = function() {}
                self._handle_lexical_decl(actual, mod_id, rel, source_bytes, nodes, edges)
            elif child.type in _IMPORT_TYPES:
                self._handle_import(child, mod_id, source_bytes, edges)

    def _handle_class(self, node, mod_id: str, rel: str, source_bytes: bytes,
                      nodes: list[CodeNode], edges: list[Edge]) -> None:
        name_node = _find_child_by_type(node, "type_identifier", "identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        cls_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        nodes.append(CodeNode(
            id=cls_id, name=name, node_type=NodeType.CLASS, file_path=rel,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            docstring=doc, parent_id=mod_id,
        ))
        edges.append(Edge(source=mod_id, target=cls_id, edge_type=EdgeType.CONTAINS))

        # Heritage: extends / implements
        heritage = _find_child_by_type(node, "class_heritage")
        if heritage:
            # JS grammar: heritage children are keyword tokens and identifiers directly
            # e.g. [extends, identifier] or [extends, identifier, implements, type_identifier, ...]
            mode = None
            for clause in heritage.children:
                if clause.type == "extends":
                    mode = "extends"
                elif clause.type == "implements":
                    mode = "implements"
                elif clause.type == "extends_clause":
                    base = _find_child_by_type(clause, "identifier", "member_expression", "type_identifier")
                    if base:
                        edges.append(Edge(source=cls_id, target=_node_text(base, source_bytes),
                                          edge_type=EdgeType.INHERITS))
                elif clause.type == "implements_clause":
                    for iface in _find_children_by_type(clause, "type_identifier", "generic_type"):
                        name_n = iface if iface.type == "type_identifier" else _find_child_by_type(iface, "type_identifier")
                        if name_n:
                            edges.append(Edge(source=cls_id, target=_node_text(name_n, source_bytes),
                                              edge_type=EdgeType.IMPLEMENTS))
                elif clause.type in ("identifier", "member_expression", "type_identifier"):
                    if mode == "extends":
                        edges.append(Edge(source=cls_id, target=_node_text(clause, source_bytes),
                                          edge_type=EdgeType.INHERITS))
                    elif mode == "implements":
                        edges.append(Edge(source=cls_id, target=_node_text(clause, source_bytes),
                                          edge_type=EdgeType.IMPLEMENTS))

        # Methods
        body = _find_child_by_type(node, "class_body")
        if body:
            for item in body.children:
                if item.type in _METHOD_TYPES:
                    self._handle_method(item, cls_id, name, rel, source_bytes, nodes, edges)

    def _handle_method(self, node, cls_id: str, cls_name: str, rel: str,
                       source_bytes: bytes, nodes: list[CodeNode], edges: list[Edge]) -> None:
        name_node = _find_child_by_type(node, "property_identifier", "computed_property_name")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        meth_id = f"{rel}::{cls_name}.{name}"
        doc = _get_jsdoc(node, source_bytes)

        # Build signature
        params_node = _find_child_by_type(node, "formal_parameters")
        sig = f"{name}({_node_text(params_node, source_bytes) if params_node else ''})"

        nodes.append(CodeNode(
            id=meth_id, name=f"{cls_name}.{name}", node_type=NodeType.METHOD,
            file_path=rel, line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=sig, docstring=doc, parent_id=cls_id,
        ))
        edges.append(Edge(source=cls_id, target=meth_id, edge_type=EdgeType.CONTAINS))
        self._extract_calls(node, meth_id, source_bytes, edges)

    def _handle_function(self, node, mod_id: str, rel: str, source_bytes: bytes,
                         nodes: list[CodeNode], edges: list[Edge]) -> None:
        name_node = _find_child_by_type(node, "identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        func_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        params_node = _find_child_by_type(node, "formal_parameters")
        sig = f"function {name}({_node_text(params_node, source_bytes) if params_node else ''})"

        nodes.append(CodeNode(
            id=func_id, name=name, node_type=NodeType.FUNCTION, file_path=rel,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=sig, docstring=doc, parent_id=mod_id,
        ))
        edges.append(Edge(source=mod_id, target=func_id, edge_type=EdgeType.CONTAINS))
        self._extract_calls(node, func_id, source_bytes, edges)

    def _handle_interface(self, node, mod_id: str, rel: str, source_bytes: bytes,
                          nodes: list[CodeNode], edges: list[Edge]) -> None:
        name_node = _find_child_by_type(node, "type_identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        iface_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        nodes.append(CodeNode(
            id=iface_id, name=name, node_type=NodeType.INTERFACE, file_path=rel,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            docstring=doc, parent_id=mod_id,
        ))
        edges.append(Edge(source=mod_id, target=iface_id, edge_type=EdgeType.CONTAINS))

        # extends clause
        extends = _find_child_by_type(node, "extends_type_clause")
        if extends:
            for tid in _find_children_by_type(extends, "type_identifier"):
                edges.append(Edge(source=iface_id, target=_node_text(tid, source_bytes),
                                  edge_type=EdgeType.INHERITS))

    def _handle_enum(self, node, mod_id: str, rel: str, source_bytes: bytes,
                     nodes: list[CodeNode], edges: list[Edge]) -> None:
        name_node = _find_child_by_type(node, "identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        enum_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        nodes.append(CodeNode(
            id=enum_id, name=name, node_type=NodeType.ENUM, file_path=rel,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            docstring=doc, parent_id=mod_id,
        ))
        edges.append(Edge(source=mod_id, target=enum_id, edge_type=EdgeType.CONTAINS))

    def _handle_lexical_decl(self, node, mod_id: str, rel: str, source_bytes: bytes,
                             nodes: list[CodeNode], edges: list[Edge]) -> None:
        """Handle `const foo = () => {}` or `const foo = function() {}`."""
        for decl in _find_children_by_type(node, "variable_declarator"):
            name_node = _find_child_by_type(decl, "identifier")
            value_node = _find_child_by_type(decl, "arrow_function", "function_expression", "function")
            if name_node and value_node:
                name = _node_text(name_node, source_bytes)
                func_id = f"{rel}::{name}"
                doc = _get_jsdoc(node, source_bytes)

                params_node = _find_child_by_type(value_node, "formal_parameters")
                sig = f"const {name} = ({_node_text(params_node, source_bytes) if params_node else ''}) =>"

                nodes.append(CodeNode(
                    id=func_id, name=name, node_type=NodeType.FUNCTION, file_path=rel,
                    line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                    signature=sig, docstring=doc, parent_id=mod_id,
                ))
                edges.append(Edge(source=mod_id, target=func_id, edge_type=EdgeType.CONTAINS))
                self._extract_calls(value_node, func_id, source_bytes, edges)

    def _handle_import(self, node, mod_id: str, source_bytes: bytes, edges: list[Edge]) -> None:
        source_node = _find_child_by_type(node, "string")
        if source_node:
            target = _node_text(source_node, source_bytes).strip("'\"")
            edges.append(Edge(source=mod_id, target=target, edge_type=EdgeType.IMPORTS))

    def _extract_calls(self, node, caller_id: str, source_bytes: bytes, edges: list[Edge]) -> None:
        """Walk subtree to find call expressions."""
        if node is None:
            return
        cursor = node.walk()
        reached_root = False
        while not reached_root:
            current = cursor.node
            if current.type == "call_expression":
                func_node = current.children[0] if current.children else None
                if func_node:
                    name = _node_text(func_node, source_bytes)
                    # Trim to just the function name (not args)
                    if name and not name.startswith("("):
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
