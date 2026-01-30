"""Tests for the grep tool in base_tools."""

from __future__ import annotations

from pathlib import Path

from coding_agent.services.file_service import FileService
from coding_agent.tools.base_tools import create_base_tools


class FakeFileService(FileService):
    """Minimal FileService backed by a real temp directory."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def read_file(self, path: str) -> str:
        return (self.work_dir / path).read_text()

    def write_file(self, path: str, content: str) -> None:
        (self.work_dir / path).write_text(content)

    def edit_file(self, path: str, old_text: str, new_text: str) -> None:
        p = self.work_dir / path
        p.write_text(p.read_text().replace(old_text, new_text, 1))

    def list_directory(self, path: str = ".") -> list[str]:
        return [e.name for e in (self.work_dir / path).iterdir()]

    def file_exists(self, path: str) -> bool:
        return (self.work_dir / path).exists()


def _get_grep(service: FakeFileService):
    tools = create_base_tools(service)
    return next(t for t in tools if t.name == "grep")


def _setup(tmp_path: Path) -> FakeFileService:
    """Create a sample project structure."""
    (tmp_path / "app.py").write_text(
        "class Validator:\n"
        "    def validate(self, data):\n"
        "        pass\n"
        "\n"
        "def helper():\n"
        "    return 42\n"
    )
    (tmp_path / "utils.py").write_text(
        "import os\n"
        "def helper():\n"
        "    return os.getcwd()\n"
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "mod.py").write_text("# empty module\n")
    (sub / "data.txt").write_text("some data line\nanother line\n")
    return FakeFileService(tmp_path)


class TestGrepDirectory:
    def test_basic_search(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "def helper"})
        assert "app.py:5: def helper():" in result
        assert "utils.py:2: def helper():" in result

    def test_regex_pattern(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": r"class \w+"})
        assert "app.py:1:" in result
        assert "Validator" in result

    def test_include_filter(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "line", "include": "*.txt"})
        assert "data.txt" in result
        assert ".py" not in result

    def test_no_matches(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "nonexistent_xyz"})
        assert "No matches" in result

    def test_subdirectory_path(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "empty", "path": "sub"})
        assert "sub/mod.py:1:" in result

    def test_invalid_regex_falls_back_to_literal(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "def helper("})
        assert "No matches" not in result
        assert "def helper()" in result


class TestGrepSingleFile:
    """Tests for the bug fix: grep on a single file path."""

    def test_grep_single_file(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "def", "path": "app.py"})
        assert "app.py:2:" in result
        assert "app.py:5:" in result

    def test_grep_single_file_with_include(self, tmp_path: Path):
        """include filter is irrelevant for single file, should still work."""
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "class", "path": "app.py", "include": "*.py"})
        assert "Validator" in result

    def test_grep_single_file_no_match(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "zzz_nothing", "path": "app.py"})
        assert "No matches" in result

    def test_grep_single_file_in_subdir(self, tmp_path: Path):
        svc = _setup(tmp_path)
        grep = _get_grep(svc)
        result = grep.execute({"pattern": "data", "path": "sub/data.txt"})
        assert "sub/data.txt:1:" in result
