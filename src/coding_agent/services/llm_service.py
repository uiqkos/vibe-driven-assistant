from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from coding_agent.config import ModelConfig, get_model_config, settings


def _create_openai_client(base_url: str = "") -> OpenAI:
    """Create an OpenAI client, optionally wrapped with PromptLayer."""
    url = base_url or settings.llm_base_url
    if settings.promptlayer_api_key:
        from promptlayer import PromptLayer

        promptlayer_client = PromptLayer(api_key=settings.promptlayer_api_key)
        return promptlayer_client.openai.OpenAI(
            api_key=settings.llm_api_key,
            base_url=url,
        )
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=url,
    )


class LLMService:
    def __init__(self, config: ModelConfig | None = None) -> None:
        if config is None:
            config = get_model_config()
        self._config = config
        self.client = _create_openai_client(config.base_url)
        self.model = config.model or settings.llm_model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    def _get_temperature(self) -> float:
        return self._temperature if self._temperature is not None else 0.2

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self._get_temperature(),
            "pl_tags": ["coding-agent", "generate"],
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def generate_with_tools(self, messages: list[dict], tools: list[dict]):
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self._get_temperature(),
        }
        if tools:
            kwargs["tools"] = tools
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        kwargs["pl_tags"] = ["coding-agent", "generate-with-tools"]
        return self.client.chat.completions.create(**kwargs)
