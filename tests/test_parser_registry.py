"""Tests for ParserRegistry and BaseParser integration."""

from pathlib import Path

from coding_agent.code_indexer.parser.ast_parser import PythonAstParser
from coding_agent.code_indexer.parser.base_parser import BaseParser
from coding_agent.code_indexer.parser.registry import ParserRegistry


def test_register_and_lookup():
    registry = ParserRegistry()
    parser = PythonAstParser()
    registry.register(parser)

    assert registry.get_parser(Path("foo.py")) is parser
    assert registry.get_parser(Path("bar.txt")) is None


def test_get_all_extensions():
    registry = ParserRegistry()
    registry.register(PythonAstParser())
    exts = registry.get_all_extensions()
    assert ".py" in exts


def test_python_parser_is_base_parser():
    parser = PythonAstParser()
    assert isinstance(parser, BaseParser)
    assert ".py" in parser.get_supported_extensions()


def test_python_parser_parses_file(tmp_path):
    f = tmp_path / "example.py"
    f.write_text("def hello():\n    pass\n")
    parser = PythonAstParser()
    nodes, edges = parser.parse_file(f, tmp_path)
    assert len(nodes) >= 2  # module + function
    names = {n.name for n in nodes}
    assert "hello" in names


def test_multiple_parsers():
    registry = ParserRegistry()
    py_parser = PythonAstParser()
    registry.register(py_parser)

    assert registry.get_parser(Path("a.py")) is py_parser
    assert registry.get_parser(Path("a.js")) is None
    assert len(registry.get_all_extensions()) == 1
