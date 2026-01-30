from __future__ import annotations

import logging
from pathlib import Path

from coding_agent.models.schemas import FileChange
from coding_agent.services.file_service import FileService

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_SIZE_KB = 100


class LocalService(FileService):
    def __init__(
        self,
        work_dir: Path | None = None,
        max_file_size_kb: int = DEFAULT_MAX_FILE_SIZE_KB,
    ) -> None:
        self.work_dir = (work_dir or Path.cwd()).resolve()
        self.max_file_size_kb = max_file_size_kb

    def read_paths(self, paths: list[str]) -> str:
        logger.info("Reading paths: %s", paths)
        parts: list[str] = []
        for p in paths:
            path = Path(p)
            if path.is_file():
                self._read_file(path, parts)
            elif path.is_dir():
                logger.info("Scanning directory: %s", path)
                py_files = sorted(path.rglob("*.py"))
                logger.info("Found %d .py files in %s", len(py_files), path)
                for fp in py_files:
                    self._read_file(fp, parts)
            else:
                logger.warning("Path not found: %s", path)
        logger.info("Total files in context: %d", len(parts))
        return "\n".join(parts)

    def _read_file(self, path: Path, parts: list[str]) -> None:
        size = path.stat().st_size
        if size > self.max_file_size_kb * 1024:
            logger.warning("Skipping %s (%.1f KB > %d KB limit)", path, size / 1024, self.max_file_size_kb)
            return
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Cannot read %s: %s", path, e)
            return
        logger.info("Read file: %s (%d lines, %.1f KB)", path, content.count("\n") + 1, size / 1024)
        parts.append(f'<file path="{path}">\n{content}\n</file>')

    def write_solution(self, files: list[FileChange]) -> None:
        for f in files:
            p = Path(f.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f.content, encoding="utf-8")
            logger.info("Written: %s (%d lines)", f.path, f.content.count("\n") + 1)

    def _resolve(self, path: str) -> Path:
        """Resolve path relative to work_dir."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.work_dir / p

    # --- FileService interface ---

    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def edit_file(self, path: str, old_text: str, new_text: str) -> None:
        p = self._resolve(path)
        content = p.read_text(encoding="utf-8")
        if old_text not in content:
            raise ValueError(f"old_text not found in {path}")
        content = content.replace(old_text, new_text, 1)
        p.write_text(content, encoding="utf-8")

    def list_directory(self, path: str = ".") -> list[str]:
        p = self._resolve(path)
        return sorted(str(entry.relative_to(p)) for entry in p.iterdir())

    def file_exists(self, path: str) -> bool:
        return self._resolve(path).exists()
