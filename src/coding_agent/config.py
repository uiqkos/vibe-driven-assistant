from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    github_token: str = ""
    llm_api_key: str = ""
    promptlayer_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"
    max_context_files: int = 10
    max_file_size_kb: int = 50
    max_iterations: int = 3

    # Code indexer settings
    indexer_exclude_patterns: list[str] = [
        "__pycache__", ".git", ".venv", "venv", "node_modules", ".tox", "*.egg-info",
    ]
    indexer_batch_size: int = 10

    # Workspace
    workdir: str = "~/.coding-agent"

    # Agentic settings
    agentic_max_steps: int = 30

    # RAG settings
    rag_code_embedding_model: str = "mistralai/codestral-embed-2505"
    rag_summary_embedding_model: str = "openai/text-embedding-3-small"
    rag_chunk_size: int = 3000
    rag_chunk_overlap: int = 1000
    rag_top_k: int = 10

    # GitHub App settings
    github_app_id: int = 0
    github_app_private_key_path: str = ""
    github_app_webhook_secret: str = ""


settings = Settings()


@dataclass
class ModelConfig:
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    base_url: str = ""


_models_config_cache: dict | None = None


def _load_models_yaml() -> dict:
    global _models_config_cache
    if _models_config_cache is not None:
        return _models_config_cache

    config_path = os.environ.get("MODELS_CONFIG_PATH", "models.yaml")
    path = Path(config_path)
    if not path.is_file():
        _models_config_cache = {}
        return _models_config_cache

    import yaml

    with open(path) as f:
        _models_config_cache = yaml.safe_load(f) or {}
    return _models_config_cache


def get_model_config(agent_name: str = "") -> ModelConfig:
    """Get model config for an agent, merging default + agent override.

    Falls back to Settings env variables if models.yaml doesn't exist.
    """
    data = _load_models_yaml()

    if not data:
        # No YAML config — use env-based Settings as fallback
        return ModelConfig(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )

    # Start with default section
    default = data.get("default", {})
    merged = {
        "model": default.get("model", settings.llm_model),
        "temperature": default.get("temperature"),
        "max_tokens": default.get("max_tokens"),
        "base_url": default.get("base_url", settings.llm_base_url),
    }

    # Merge agent-specific overrides
    if agent_name:
        agents = data.get("agents", {})
        agent_override = agents.get(agent_name, {})
        for key, value in agent_override.items():
            if key in merged:
                merged[key] = value

    return ModelConfig(
        model=merged["model"],
        temperature=merged["temperature"],
        max_tokens=merged["max_tokens"],
        base_url=merged["base_url"],
    )
