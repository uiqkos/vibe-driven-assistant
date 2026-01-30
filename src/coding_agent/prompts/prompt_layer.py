"""Prompt layer — loads prompt templates from .txt files in templates/ directory."""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without extension).

    Returns the raw template string with {variable} placeholders.
    """
    if name not in _cache:
        path = TEMPLATES_DIR / f"{name}.txt"
        _cache[name] = path.read_text(encoding="utf-8").strip()
    return _cache[name]


def render_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template and fill in variables."""
    return load_prompt(name).format(**kwargs)
