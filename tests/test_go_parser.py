"""Tests for the Go parser."""

from pathlib import Path

import pytest

try:
    from coding_agent.code_indexer.parser.go_parser import GoParser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter Go not installed")


def test_supported_extensions():
    parser = GoParser()
    assert parser.get_supported_extensions() == [".go"]


def test_parse_go_file():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    names = {n.name for n in nodes}
    assert "NewUser" in names
    assert "User" in names
    assert "Admin" in names


def test_parse_go_structs():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    structs = [n for n in nodes if n.node_type == NodeType.STRUCT]
    struct_names = {s.name for s in structs}
    assert "User" in struct_names
    assert "Admin" in struct_names


def test_parse_go_interfaces():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    interfaces = [n for n in nodes if n.node_type == NodeType.INTERFACE]
    iface_names = {i.name for i in interfaces}
    assert "Reader" in iface_names
    assert "Writer" in iface_names
    assert "ReadWriter" in iface_names


def test_parse_go_functions():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    functions = [n for n in nodes if n.node_type == NodeType.FUNCTION]
    func_names = {f.name for f in functions}
    assert "NewUser" in func_names


def test_parse_go_methods():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    from coding_agent.code_indexer.models import NodeType
    methods = [n for n in nodes if n.node_type == NodeType.METHOD]
    method_names = {m.name for m in methods}
    assert "User.FullName" in method_names
    assert "User.Greet" in method_names
    assert "User.Promote" in method_names


def test_parse_go_imports():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    from coding_agent.code_indexer.models import EdgeType
    imports = [e for e in edges if e.edge_type == EdgeType.IMPORTS]
    targets = {e.target for e in imports}
    assert "fmt" in targets
    assert "strings" in targets


def test_parse_go_embedded_struct():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    from coding_agent.code_indexer.models import EdgeType
    inherits = [e for e in edges if e.edge_type == EdgeType.INHERITS]
    # Admin embeds User
    assert any(e.source.endswith("::Admin") and e.target == "User" for e in inherits)


def test_parse_go_embedded_interface():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    from coding_agent.code_indexer.models import EdgeType
    inherits = [e for e in edges if e.edge_type == EdgeType.INHERITS]
    # ReadWriter embeds Reader and Writer
    assert any(e.source.endswith("::ReadWriter") for e in inherits)


def test_parse_go_doc_comments():
    parser = GoParser()
    nodes, edges = parser.parse_file(FIXTURES / "sample.go", FIXTURES)
    new_user = next((n for n in nodes if n.name == "NewUser"), None)
    assert new_user is not None
    from coding_agent.code_indexer.models import NodeType
    assert new_user.node_type == NodeType.FUNCTION
