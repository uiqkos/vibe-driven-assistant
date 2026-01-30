"""ChromaDB stores for code chunks and summaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from coding_agent.rag.config import RAGConfig
from coding_agent.rag.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class _OpenRouterEmbeddingFunction(chromadb.EmbeddingFunction):
    """ChromaDB embedding function backed by OpenRouter."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._client.embed(input)


@dataclass
class SearchResult:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float


def _dedup(
    ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Remove duplicate IDs, keeping the last occurrence."""
    seen: dict[str, int] = {}
    for i, id_ in enumerate(ids):
        seen[id_] = i  # last wins
    if len(seen) == len(ids):
        return ids, documents, metadatas
    indices = sorted(seen.values())
    logger.warning("Deduped %d -> %d entries (%d duplicates removed)",
                   len(ids), len(indices), len(ids) - len(indices))
    return (
        [ids[i] for i in indices],
        [documents[i] for i in indices],
        [metadatas[i] for i in indices],
    )


class CodeStore:
    """ChromaDB collection for code chunks."""

    def __init__(self, persist_dir: Path, config: RAGConfig) -> None:
        self._config = config
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._emb_client = EmbeddingClient(config.code_embedding_model, config.embedding_batch_size)
        self._ef = _OpenRouterEmbeddingFunction(self._emb_client)
        self._collection = self._client.get_or_create_collection(
            name="code_chunks",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Add chunks to the store. Handles batching internally."""
        if not ids:
            return
        ids, documents, metadatas = _dedup(ids, documents, metadatas)
        logger.info("CodeStore: upserting %d chunks", len(ids))
        batch = 500
        for i in range(0, len(ids), batch):
            batch_end = min(i + batch, len(ids))
            logger.debug("CodeStore: upserting batch %d-%d / %d", i, batch_end, len(ids))
            self._collection.upsert(
                ids=ids[i : batch_end],
                documents=documents[i : batch_end],
                metadatas=metadatas[i : batch_end],
            )
        logger.info("CodeStore: upsert complete, total count=%d", self._collection.count())

    def query(self, text: str, top_k: int = 10, where: dict | None = None) -> list[SearchResult]:
        logger.debug("CodeStore: querying top_k=%d, where=%s, query='%s'", top_k, where, text[:80])
        kwargs: dict[str, Any] = {"query_texts": [text], "n_results": top_k}
        if where:
            kwargs["where"] = where
        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            logger.error("Code store query failed: %s", e)
            return []
        parsed = self._parse_results(results)
        logger.debug("CodeStore: returned %d results", len(parsed))
        return parsed

    def delete_by_file(self, file_paths: list[str]) -> None:
        """Delete all chunks belonging to given file paths."""
        logger.info("CodeStore: deleting chunks for %d files", len(file_paths))
        for fp in file_paths:
            try:
                self._collection.delete(where={"file_path": fp})
                logger.debug("CodeStore: deleted chunks for %s", fp)
            except Exception:
                logger.debug("CodeStore: no chunks to delete for %s", fp)

    def clear(self) -> None:
        logger.info("CodeStore: clearing collection")
        self._client.delete_collection("code_chunks")
        self._collection = self._client.get_or_create_collection(
            name="code_chunks",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("CodeStore: collection cleared")

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _parse_results(results: dict) -> list[SearchResult]:
        out: list[SearchResult] = []
        if not results or not results.get("ids"):
            return out
        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(SearchResult(
                id=id_,
                content=doc,
                metadata=meta or {},
                score=round(1.0 - dist, 4),  # cosine distance → similarity
            ))
        return out


class SummaryStore:
    """ChromaDB collection for code summaries."""

    def __init__(self, persist_dir: Path, config: RAGConfig) -> None:
        self._config = config
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._emb_client = EmbeddingClient(config.summary_embedding_model, config.embedding_batch_size)
        self._ef = _OpenRouterEmbeddingFunction(self._emb_client)
        self._collection = self._client.get_or_create_collection(
            name="summaries",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        ids, documents, metadatas = _dedup(ids, documents, metadatas)
        logger.info("SummaryStore: upserting %d summaries", len(ids))
        batch = 500
        for i in range(0, len(ids), batch):
            batch_end = min(i + batch, len(ids))
            logger.debug("SummaryStore: upserting batch %d-%d / %d", i, batch_end, len(ids))
            self._collection.upsert(
                ids=ids[i : batch_end],
                documents=documents[i : batch_end],
                metadatas=metadatas[i : batch_end],
            )
        logger.info("SummaryStore: upsert complete, total count=%d", self._collection.count())

    def query(self, text: str, top_k: int = 10) -> list[SearchResult]:
        logger.debug("SummaryStore: querying top_k=%d, query='%s'", top_k, text[:80])
        try:
            results = self._collection.query(query_texts=[text], n_results=top_k)
        except Exception as e:
            logger.error("Summary store query failed: %s", e)
            return []
        parsed = CodeStore._parse_results(results)
        logger.debug("SummaryStore: returned %d results", len(parsed))
        return parsed

    def delete_by_file(self, file_paths: list[str]) -> None:
        logger.info("SummaryStore: deleting summaries for %d files", len(file_paths))
        for fp in file_paths:
            try:
                self._collection.delete(where={"file_path": fp})
                logger.debug("SummaryStore: deleted summaries for %s", fp)
            except Exception:
                logger.debug("SummaryStore: no summaries to delete for %s", fp)

    def clear(self) -> None:
        logger.info("SummaryStore: clearing collection")
        self._client.delete_collection("summaries")
        self._collection = self._client.get_or_create_collection(
            name="summaries",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("SummaryStore: collection cleared")

    def count(self) -> int:
        return self._collection.count()
