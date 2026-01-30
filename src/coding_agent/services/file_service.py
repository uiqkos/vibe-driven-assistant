"""Abstract file service interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FileService(ABC):
    @abstractmethod
    def read_file(self, path: str) -> str: ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None: ...

    @abstractmethod
    def edit_file(self, path: str, old_text: str, new_text: str) -> None: ...

    @abstractmethod
    def list_directory(self, path: str = ".") -> list[str]: ...

    @abstractmethod
    def file_exists(self, path: str) -> bool: ...
