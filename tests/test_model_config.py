"""Tests for models.yaml configuration loading and ModelConfig merging."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from coding_agent.config import ModelConfig, get_model_config, _load_models_yaml


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the module-level YAML cache before each test."""
    import coding_agent.config as cfg
    cfg._models_config_cache = None
    yield
    cfg._models_config_cache = None


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------

def test_model_config_defaults():
    mc = ModelConfig()
    assert mc.model == ""
    assert mc.temperature is None
    assert mc.max_tokens is None
    assert mc.base_url == ""


def test_model_config_custom_values():
    mc = ModelConfig(model="openai/gpt-4.1", temperature=0.5, max_tokens=4096, base_url="https://example.com")
    assert mc.model == "openai/gpt-4.1"
    assert mc.temperature == 0.5
    assert mc.max_tokens == 4096
    assert mc.base_url == "https://example.com"


# ---------------------------------------------------------------------------
# _load_models_yaml
# ---------------------------------------------------------------------------

def test_load_yaml_missing_file(tmp_path):
    """When file doesn't exist, returns empty dict."""
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "nope.yaml")}):
        result = _load_models_yaml()
    assert result == {}


def test_load_yaml_valid_file(tmp_path):
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(
        "default:\n"
        "  model: openai/gpt-4.1-mini\n"
        "  temperature: 0.3\n"
    )
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(yaml_file)}):
        result = _load_models_yaml()
    assert result["default"]["model"] == "openai/gpt-4.1-mini"
    assert result["default"]["temperature"] == 0.3


def test_load_yaml_empty_file(tmp_path):
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text("")
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(yaml_file)}):
        result = _load_models_yaml()
    assert result == {}


def test_load_yaml_caches_result(tmp_path):
    """Second call returns cached dict without re-reading."""
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text("default:\n  model: m1\n")
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(yaml_file)}):
        first = _load_models_yaml()
        # Modify file — should NOT affect cached result
        yaml_file.write_text("default:\n  model: m2\n")
        second = _load_models_yaml()
    assert first is second
    assert first["default"]["model"] == "m1"


# ---------------------------------------------------------------------------
# get_model_config — no YAML file (env fallback)
# ---------------------------------------------------------------------------

def test_get_model_config_no_yaml_falls_back_to_settings(tmp_path):
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "missing.yaml")}):
        mc = get_model_config()
    # Should use Settings defaults
    assert mc.model != ""  # whatever Settings.llm_model resolves to
    assert mc.temperature is None
    assert mc.max_tokens is None


def test_get_model_config_no_yaml_agent_name_ignored(tmp_path):
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "missing.yaml")}):
        mc = get_model_config("review")
    assert mc.model != ""


# ---------------------------------------------------------------------------
# get_model_config — with YAML, default section only
# ---------------------------------------------------------------------------

YAML_DEFAULT_ONLY = (
    "default:\n"
    "  model: openai/gpt-4.1-mini\n"
    "  temperature: 0.2\n"
    "  max_tokens: 8000\n"
    "  base_url: https://custom.api/v1\n"
)


def test_default_section_no_agent(tmp_path):
    (tmp_path / "m.yaml").write_text(YAML_DEFAULT_ONLY)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config()
    assert mc.model == "openai/gpt-4.1-mini"
    assert mc.temperature == 0.2
    assert mc.max_tokens == 8000
    assert mc.base_url == "https://custom.api/v1"


def test_default_section_unknown_agent(tmp_path):
    """Unknown agent name still gets default values."""
    (tmp_path / "m.yaml").write_text(YAML_DEFAULT_ONLY)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config("unknown_agent")
    assert mc.model == "openai/gpt-4.1-mini"
    assert mc.temperature == 0.2


# ---------------------------------------------------------------------------
# get_model_config — with agent overrides
# ---------------------------------------------------------------------------

YAML_WITH_AGENTS = (
    "default:\n"
    "  model: openai/gpt-4.1-mini\n"
    "  temperature: 0.2\n"
    "  base_url: https://openrouter.ai/api/v1\n"
    "agents:\n"
    "  review:\n"
    "    model: openai/gpt-4.1\n"
    "    temperature: 0.1\n"
    "  agentic:\n"
    "    model: openai/gpt-4.1\n"
    "    max_tokens: 16000\n"
)


def test_agent_override_merges_with_default(tmp_path):
    (tmp_path / "m.yaml").write_text(YAML_WITH_AGENTS)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config("review")
    assert mc.model == "openai/gpt-4.1"
    assert mc.temperature == 0.1
    # base_url inherited from default
    assert mc.base_url == "https://openrouter.ai/api/v1"
    # max_tokens not set in review or default
    assert mc.max_tokens is None


def test_agent_override_partial(tmp_path):
    """Agent overrides only specified fields; rest comes from default."""
    (tmp_path / "m.yaml").write_text(YAML_WITH_AGENTS)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config("agentic")
    assert mc.model == "openai/gpt-4.1"
    assert mc.max_tokens == 16000
    # temperature inherited from default
    assert mc.temperature == 0.2


def test_agent_not_overridden_gets_default(tmp_path):
    (tmp_path / "m.yaml").write_text(YAML_WITH_AGENTS)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config("solve")
    assert mc.model == "openai/gpt-4.1-mini"
    assert mc.temperature == 0.2


# ---------------------------------------------------------------------------
# get_model_config — null handling in YAML
# ---------------------------------------------------------------------------

def test_null_max_tokens_in_yaml(tmp_path):
    yaml_content = (
        "default:\n"
        "  model: m\n"
        "  max_tokens: null\n"
    )
    (tmp_path / "m.yaml").write_text(yaml_content)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config()
    assert mc.max_tokens is None


def test_null_temperature_in_yaml(tmp_path):
    yaml_content = (
        "default:\n"
        "  model: m\n"
        "  temperature: null\n"
    )
    (tmp_path / "m.yaml").write_text(yaml_content)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config()
    assert mc.temperature is None


# ---------------------------------------------------------------------------
# get_model_config — minimal YAML (missing optional fields)
# ---------------------------------------------------------------------------

def test_minimal_yaml_only_model(tmp_path):
    (tmp_path / "m.yaml").write_text("default:\n  model: my-model\n")
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config()
    assert mc.model == "my-model"
    assert mc.temperature is None
    assert mc.max_tokens is None


def test_yaml_no_default_section(tmp_path):
    """YAML with only agents section — default fields come from Settings."""
    yaml_content = (
        "agents:\n"
        "  review:\n"
        "    model: openai/gpt-4.1\n"
    )
    (tmp_path / "m.yaml").write_text(yaml_content)
    with patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "m.yaml")}):
        mc = get_model_config("review")
    assert mc.model == "openai/gpt-4.1"


# ---------------------------------------------------------------------------
# LLMService integration with ModelConfig
# ---------------------------------------------------------------------------

def test_llm_service_uses_model_config():
    """LLMService picks up model/temperature/max_tokens from ModelConfig."""
    from coding_agent.services.llm_service import LLMService

    cfg = ModelConfig(model="test-model", temperature=0.7, max_tokens=512, base_url="https://example.com/v1")
    svc = LLMService(config=cfg)
    assert svc.model == "test-model"
    assert svc._temperature == 0.7
    assert svc._max_tokens == 512


def test_llm_service_default_temperature():
    """When temperature is None, _get_temperature falls back to 0.2."""
    from coding_agent.services.llm_service import LLMService

    cfg = ModelConfig(model="m", base_url="https://example.com/v1")
    svc = LLMService(config=cfg)
    assert svc._get_temperature() == 0.2


def test_llm_service_custom_temperature():
    from coding_agent.services.llm_service import LLMService

    cfg = ModelConfig(model="m", temperature=0.9, base_url="https://example.com/v1")
    svc = LLMService(config=cfg)
    assert svc._get_temperature() == 0.9


def test_llm_service_zero_temperature():
    from coding_agent.services.llm_service import LLMService

    cfg = ModelConfig(model="m", temperature=0.0, base_url="https://example.com/v1")
    svc = LLMService(config=cfg)
    assert svc._get_temperature() == 0.0
