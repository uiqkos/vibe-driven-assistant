"""OpenRouter-compatible embedding client."""

from __future__ import annotations

import logging

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from coding_agent.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Wrapper over OpenAI-compatible API for embeddings via OpenRouter."""

    def __init__(self, model: str, batch_size: int = 50, batch_max_chars: int = 80_000) -> None:
        self.model = model
        self.batch_size = batch_size
        self.batch_max_chars = batch_max_chars
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)),
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Replace empty/whitespace-only texts with a placeholder
        sanitized = [t if t.strip() else " " for t in texts]
        total_chars = sum(len(t) for t in sanitized)
        logger.info("Embedding batch: %d texts, %d total chars, model=%s",
                     len(sanitized), total_chars, self.model)
        response = self._client.embeddings.create(model=self.model, input=sanitized)
        if not response.data:
            logger.error(
                "Empty embedding response: model=%s, input_count=%d, total_chars=%d, response=%s",
                self.model, len(sanitized), total_chars, response,
            )
            raise ValueError(f"No embedding data received for {len(sanitized)} texts ({total_chars} chars)")
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        logger.debug("Batch embedded: %d vectors of dim %d", len(embeddings),
                      len(embeddings[0]) if embeddings else 0)
        return embeddings

    def _split_batches(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches respecting both count and char limits."""
        batches: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for t in texts:
            t_len = len(t)
            if current and (len(current) >= self.batch_size or current_chars + t_len > self.batch_max_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(t)
            current_chars += t_len
        if current:
            batches.append(current)
        return batches

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a list of texts, batching as needed.

        Returns a list aligned with *texts*. Failed batches produce ``None``
        entries so the caller can skip them while keeping the rest.
        """
        if not texts:
            return []
        batches = self._split_batches(texts)
        total_chars = sum(len(t) for t in texts)
        logger.info("Embedding %d texts (%d chars) in %d batches (model=%s, batch_size=%d, max_chars=%d)",
                     len(texts), total_chars, len(batches), self.model, self.batch_size, self.batch_max_chars)
        all_embeddings: list[list[float] | None] = []
        failed_batches = 0
        for batch_num, batch in enumerate(batches, 1):
            logger.info("Embedding batch %d/%d (%d texts, %d chars)...",
                         batch_num, len(batches), len(batch), sum(len(t) for t in batch))
            try:
                all_embeddings.extend(self._embed_batch(batch))
            except Exception:
                logger.exception("Embedding batch %d/%d failed, skipping %d texts",
                                 batch_num, len(batches), len(batch))
                all_embeddings.extend([None] * len(batch))
                failed_batches += 1
        logger.info("Embedding complete: %d vectors (%d batches failed)",
                     sum(1 for e in all_embeddings if e is not None), failed_batches)
        return all_embeddings
