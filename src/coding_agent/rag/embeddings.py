"""OpenRouter-compatible embedding client."""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from coding_agent.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Wrapper over OpenAI-compatible API for embeddings via OpenRouter."""

    def __init__(self, model: str, batch_size: int = 100) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        logger.debug("Embedding batch of %d texts with model %s", len(texts), self.model)
        response = self._client.embeddings.create(model=self.model, input=texts)
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        logger.debug("Batch embedded: %d vectors of dim %d", len(embeddings),
                      len(embeddings[0]) if embeddings else 0)
        return embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching as needed."""
        if not texts:
            return []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.info("Embedding %d texts in %d batches (model=%s, batch_size=%d)",
                     len(texts), total_batches, self.model, self.batch_size)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = texts[i : i + self.batch_size]
            logger.info("Embedding batch %d/%d (%d texts)...", batch_num, total_batches, len(batch))
            all_embeddings.extend(self._embed_batch(batch))
        logger.info("Embedding complete: %d vectors", len(all_embeddings))
        return all_embeddings
