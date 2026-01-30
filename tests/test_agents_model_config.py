"""Tests that agents and CLI correctly pass ModelConfig to LLMService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coding_agent.config import ModelConfig


@pytest.fixture(autouse=True)
def _reset_config_cache():
    import coding_agent.config as cfg
    cfg._models_config_cache = None
    yield
    cfg._models_config_cache = None


# ---------------------------------------------------------------------------
# Helper: YAML that sets different models per agent
# ---------------------------------------------------------------------------

YAML_ALL_AGENTS = (
    "default:\n"
    "  model: default-model\n"
    "  temperature: 0.2\n"
    "  base_url: https://default.api/v1\n"
    "agents:\n"
    "  solve:\n"
    "    model: solve-model\n"
    "    temperature: 0.3\n"
    "  review:\n"
    "    model: review-model\n"
    "    temperature: 0.1\n"
    "  iterate:\n"
    "    model: iterate-model\n"
    "  agentic:\n"
    "    model: agentic-model\n"
    "    max_tokens: 16000\n"
    "  indexer:\n"
    "    model: indexer-model\n"
)


def _write_yaml(tmp_path, content=YAML_ALL_AGENTS):
    p = tmp_path / "models.yaml"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# CLI _build_agent (agentic) — uses lazy imports, so we patch get_model_config
# ---------------------------------------------------------------------------

class TestBuildAgent:
    def test_build_agent_passes_agentic_config(self, tmp_path):
        yaml_path = _write_yaml(tmp_path)
        work_dir = tmp_path / "repo"
        work_dir.mkdir()
        index_dir = tmp_path / "idx"  # doesn't exist — no indexer tools

        captured = {}

        original_llm_init = None
        from coding_agent.services.llm_service import LLMService as RealLLMService
        original_llm_init = RealLLMService.__init__

        def spy_init(self, config=None):
            captured["config"] = config
            # Don't actually init (needs real API key)

        with (
            patch.dict("os.environ", {"MODELS_CONFIG_PATH": yaml_path}),
            patch.object(RealLLMService, "__init__", spy_init),
            patch("coding_agent.agents.agentic_agent.AgenticAgent"),
            patch("coding_agent.agents.console_callback.ConsoleCallback"),
        ):
            from coding_agent.cli import _build_agent
            _build_agent(work_dir, index_dir)

        assert captured["config"].model == "agentic-model"
        assert captured["config"].max_tokens == 16000


# ---------------------------------------------------------------------------
# Indexer CLI _get_llm_service
# ---------------------------------------------------------------------------

class TestIndexerCli:
    def test_indexer_cli_uses_indexer_config(self, tmp_path):
        yaml_path = _write_yaml(tmp_path)
        captured = {}

        from coding_agent.services.llm_service import LLMService as RealLLMService

        def spy_init(self, config=None):
            captured["config"] = config

        with (
            patch.dict("os.environ", {"MODELS_CONFIG_PATH": yaml_path}),
            patch.object(RealLLMService, "__init__", spy_init),
        ):
            from coding_agent.code_indexer.cli import _get_llm_service
            _get_llm_service()

        assert captured["config"].model == "indexer-model"


# ---------------------------------------------------------------------------
# LLMService: generate() passes temperature & max_tokens correctly
# ---------------------------------------------------------------------------

class TestLLMServiceGenerate:
    def _make_service(self, **kwargs):
        cfg = ModelConfig(
            model=kwargs.get("model", "m"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            base_url=kwargs.get("base_url", "https://x.com/v1"),
        )
        with patch("coding_agent.services.llm_service._create_openai_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client_fn.return_value = mock_client
            from coding_agent.services.llm_service import LLMService
            svc = LLMService(config=cfg)
        return svc, mock_client

    def test_generate_uses_configured_temperature(self):
        svc, client = self._make_service(temperature=0.7)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_resp

        svc.generate("hello")
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0.7

    def test_generate_default_temperature_when_none(self):
        svc, client = self._make_service()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_resp

        svc.generate("hello")
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0.2

    def test_generate_passes_max_tokens(self):
        svc, client = self._make_service(max_tokens=1024)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_resp

        svc.generate("hello")
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["max_tokens"] == 1024

    def test_generate_omits_max_tokens_when_none(self):
        svc, client = self._make_service()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_resp

        svc.generate("hello")
        kwargs = client.chat.completions.create.call_args[1]
        assert "max_tokens" not in kwargs

    def test_generate_uses_configured_model(self):
        svc, client = self._make_service(model="my-model")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_resp

        svc.generate("hello")
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["model"] == "my-model"

    def test_generate_with_system_prompt(self):
        svc, client = self._make_service()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_resp

        svc.generate("hello", system_prompt="be nice")
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["messages"][0] == {"role": "system", "content": "be nice"}
        assert kwargs["messages"][1] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# LLMService: generate_with_tools() passes temperature & max_tokens
# ---------------------------------------------------------------------------

class TestLLMServiceGenerateWithTools:
    def _make_service(self, **kwargs):
        cfg = ModelConfig(
            model=kwargs.get("model", "m"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            base_url=kwargs.get("base_url", "https://x.com/v1"),
        )
        with patch("coding_agent.services.llm_service._create_openai_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client_fn.return_value = mock_client
            from coding_agent.services.llm_service import LLMService
            svc = LLMService(config=cfg)
        return svc, mock_client

    def test_tools_uses_configured_temperature(self):
        svc, client = self._make_service(temperature=0.5)
        svc.generate_with_tools([], [])
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0.5

    def test_tools_passes_max_tokens(self):
        svc, client = self._make_service(max_tokens=2048)
        svc.generate_with_tools([], [])
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["max_tokens"] == 2048

    def test_tools_omits_max_tokens_when_none(self):
        svc, client = self._make_service()
        svc.generate_with_tools([], [])
        kwargs = client.chat.completions.create.call_args[1]
        assert "max_tokens" not in kwargs

    def test_tools_passes_tool_definitions(self):
        svc, client = self._make_service()
        tools = [{"type": "function", "function": {"name": "foo"}}]
        svc.generate_with_tools([{"role": "user", "content": "hi"}], tools)
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["tools"] == tools

    def test_tools_omits_tools_when_empty(self):
        svc, client = self._make_service()
        svc.generate_with_tools([{"role": "user", "content": "hi"}], [])
        kwargs = client.chat.completions.create.call_args[1]
        assert "tools" not in kwargs


# ---------------------------------------------------------------------------
# LLMService: _create_openai_client receives base_url from config
# ---------------------------------------------------------------------------

class TestClientCreation:
    def test_custom_base_url_passed_to_client(self):
        cfg = ModelConfig(model="m", base_url="https://custom.example/v1")
        with patch("coding_agent.services.llm_service._create_openai_client") as mock_fn:
            mock_fn.return_value = MagicMock()
            from coding_agent.services.llm_service import LLMService
            LLMService(config=cfg)
            mock_fn.assert_called_once_with("https://custom.example/v1")

    def test_empty_base_url_passed_to_client(self):
        cfg = ModelConfig(model="m", base_url="")
        with patch("coding_agent.services.llm_service._create_openai_client") as mock_fn:
            mock_fn.return_value = MagicMock()
            from coding_agent.services.llm_service import LLMService
            LLMService(config=cfg)
            mock_fn.assert_called_once_with("")

    def test_create_openai_client_uses_base_url(self):
        import coding_agent.services.llm_service as mod
        original = mod.OpenAI
        try:
            mock_openai = MagicMock()
            mod.OpenAI = mock_openai
            # Ensure we take the non-PromptLayer branch
            with patch.object(mod.settings, "promptlayer_api_key", ""):
                mod._create_openai_client("https://custom.example/v1")
            assert mock_openai.call_args[1]["base_url"] == "https://custom.example/v1"
        finally:
            mod.OpenAI = original

    def test_create_openai_client_empty_url_falls_back(self):
        import coding_agent.services.llm_service as mod
        from coding_agent.config import settings
        original = mod.OpenAI
        try:
            mock_openai = MagicMock()
            mod.OpenAI = mock_openai
            with patch.object(mod.settings, "promptlayer_api_key", ""):
                mod._create_openai_client("")
            assert mock_openai.call_args[1]["base_url"] == settings.llm_base_url
        finally:
            mod.OpenAI = original


# ---------------------------------------------------------------------------
# LLMService: no config → uses get_model_config() default
# ---------------------------------------------------------------------------

class TestLLMServiceNoConfig:
    def test_no_config_uses_default(self, tmp_path):
        yaml_path = _write_yaml(tmp_path)
        import coding_agent.config as cfg
        cfg._models_config_cache = None

        with (
            patch.dict("os.environ", {"MODELS_CONFIG_PATH": yaml_path}),
            patch("coding_agent.services.llm_service._create_openai_client", return_value=MagicMock()),
        ):
            from coding_agent.services.llm_service import LLMService
            svc = LLMService()  # no config arg
            assert svc.model == "default-model"
            assert svc._temperature == 0.2

    def test_no_config_no_yaml_uses_settings(self, tmp_path):
        with (
            patch.dict("os.environ", {"MODELS_CONFIG_PATH": str(tmp_path / "nope.yaml")}),
            patch("coding_agent.services.llm_service._create_openai_client", return_value=MagicMock()),
        ):
            from coding_agent.services.llm_service import LLMService
            from coding_agent.config import settings
            svc = LLMService()
            assert svc.model == settings.llm_model
