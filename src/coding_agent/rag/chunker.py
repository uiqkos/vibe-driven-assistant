"""File chunking using llama_index splitters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from coding_agent.rag.config import RAGConfig

logger = logging.getLogger(__name__)

# Extension → llama_index CodeSplitter language
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".scala": "scala",
    ".swift": "swift",
    ".kt": "kotlin",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}

_TEXT_EXTENSIONS: set[str] = {".md", ".rst", ".txt", ".adoc", ".tex"}

# Binary extensions to skip
_BINARY_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".pyc", ".pyo", ".class", ".jar",
    ".db", ".sqlite", ".sqlite3",
}


@dataclass
class Chunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_index: int


@dataclass
class Chunker:
    config: RAGConfig = field(default_factory=RAGConfig)

    def _is_binary(self, path: Path) -> bool:
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return True
        try:
            with open(path, "rb") as f:
                data = f.read(8192)
            return b"\x00" in data
        except Exception:
            return True

    def _should_skip(self, path: Path, exclude_patterns: list[str]) -> bool:
        s = str(path)
        for pat in exclude_patterns:
            if pat.startswith("*"):
                if s.endswith(pat[1:]):
                    return True
            elif pat in s.split("/") or pat in s.split("\\"):
                return True
        return False

    def chunk_file(self, file_path: Path, root: Path) -> list[Chunk]:
        """Chunk a single file into pieces."""
        if self._is_binary(file_path):
            return []

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.debug("Cannot read %s", file_path)
            return []

        if not text.strip():
            return []

        rel = str(file_path.relative_to(root))
        ext = file_path.suffix.lower()
        language = _EXT_TO_LANGUAGE.get(ext, "")

        if language:
            return self._chunk_code(text, rel, language)
        elif ext in _TEXT_EXTENSIONS:
            return self._chunk_text(text, rel)
        elif ext in _EXT_TO_LANGUAGE or ext in _TEXT_EXTENSIONS:
            pass  # already handled
        else:
            # Try as plain text for unknown extensions
            return self._chunk_text(text, rel)

        return []

    def _chunk_code(self, text: str, rel_path: str, language: str) -> list[Chunk]:
        try:
            from llama_index.core.node_parser import CodeSplitter

            splitter = CodeSplitter(
                language=language,
                chunk_lines=self.config.chunk_size // 40,  # approximate chars → lines
                chunk_lines_overlap=self.config.chunk_overlap // 40,
                max_chars=self.config.chunk_size * 2,
            )
            from llama_index.core.schema import Document

            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
        except Exception:
            logger.debug("CodeSplitter failed for %s (%s), falling back to text split", rel_path, language)
            return self._chunk_text(text, rel_path, language=language)

        chunks: list[Chunk] = []
        for idx, node in enumerate(nodes):
            content = node.get_content()
            # Find line numbers
            start_line = self._find_line_number(text, content)
            end_line = start_line + content.count("\n")
            chunks.append(Chunk(
                content=content,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language,
                chunk_index=idx,
            ))
        return chunks

    def _chunk_text(self, text: str, rel_path: str, language: str = "") -> list[Chunk]:
        try:
            from llama_index.core.node_parser import SentenceSplitter

            splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            from llama_index.core.schema import Document

            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
        except Exception:
            logger.debug("SentenceSplitter failed for %s, using simple split", rel_path)
            return self._simple_split(text, rel_path, language)

        chunks: list[Chunk] = []
        for idx, node in enumerate(nodes):
            content = node.get_content()
            start_line = self._find_line_number(text, content)
            end_line = start_line + content.count("\n")
            chunks.append(Chunk(
                content=content,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language or "text",
                chunk_index=idx,
            ))
        return chunks

    def _simple_split(self, text: str, rel_path: str, language: str = "") -> list[Chunk]:
        """Fallback: split by character count."""
        chunks: list[Chunk] = []
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        idx = 0
        pos = 0
        while pos < len(text):
            end = min(pos + size, len(text))
            content = text[pos:end]
            start_line = text[:pos].count("\n") + 1
            end_line = start_line + content.count("\n")
            chunks.append(Chunk(
                content=content,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language or "text",
                chunk_index=idx,
            ))
            idx += 1
            pos += size - overlap
            if pos >= len(text):
                break
        return chunks

    def _find_line_number(self, full_text: str, chunk_text: str) -> int:
        """Find the line number where chunk_text starts in full_text."""
        idx = full_text.find(chunk_text[:200])
        if idx == -1:
            return 1
        return full_text[:idx].count("\n") + 1

    def collect_files(self, root: Path, exclude_patterns: list[str]) -> list[Path]:
        """Collect all text files under root, respecting exclude patterns."""
        logger.info("Scanning files in %s...", root)
        files: list[Path] = []
        skipped_pattern = 0
        skipped_binary = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if self._should_skip(path, exclude_patterns):
                skipped_pattern += 1
                continue
            if self._is_binary(path):
                skipped_binary += 1
                continue
            files.append(path)
        logger.info("Collected %d files (skipped: %d by pattern, %d binary)",
                     len(files), skipped_pattern, skipped_binary)
        return files
