"""Tests for the Rust parser."""

from pathlib import Path

import pytest

try:
    from coding_agent.code_indexer.parser.rust_parser import RustParser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter Rust not installed")


def test_supported_extensions():
    parser = RustParser()
    assert parser.get_supported_extensions() == [".rs"]


def test_parse_rust_file():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    names = {n.name for n in nodes}
    assert "User" in names
    assert "Status" in names
    assert "Displayable" in names
    assert "format_status" in names


def test_parse_rust_structs():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    structs = [n for n in nodes if n.node_type == NodeType.STRUCT]
    assert any(s.name == "User" for s in structs)


def test_parse_rust_enums():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    enums = [n for n in nodes if n.node_type == NodeType.ENUM]
    assert any(e.name == "Status" for e in enums)


def test_parse_rust_traits():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    traits = [n for n in nodes if n.node_type == NodeType.TRAIT]
    assert any(t.name == "Displayable" for t in traits)


def test_parse_rust_functions():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    functions = [n for n in nodes if n.node_type == NodeType.FUNCTION]
    assert any(f.name == "format_status" for f in functions)


def test_parse_rust_impl_methods():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    methods = [n for n in nodes if n.node_type == NodeType.METHOD]
    method_names = {m.name for m in methods}
    assert "User.new" in method_names
    assert "User.greet" in method_names
    assert "User.display" in method_names


def test_parse_rust_implements():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    from coding_agent.code_indexer.models import EdgeType
    implements = [e for e in edges if e.edge_type == EdgeType.IMPLEMENTS]
    assert any("Displayable" in e.target for e in implements)


def test_parse_rust_use_imports():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    from coding_agent.code_indexer.models import EdgeType
    imports = [e for e in edges if e.edge_type == EdgeType.IMPORTS]
    assert len(imports) >= 2


def test_parse_rust_doc_comments():
    parser = RustParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.rs", FIXTURES)
    user = next((n for n in nodes if n.name == "User" and n.node_type.value == "struct"), None)
    assert user is not None
