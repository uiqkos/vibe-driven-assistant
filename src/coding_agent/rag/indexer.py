"""RAG indexer — builds and updates ChromaDB indexes."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from coding_agent.config import settings
from coding_agent.rag.chunker import Chunker
from coding_agent.rag.config import RAGConfig
from coding_agent.rag.stores import CodeStore, SummaryStore

logger = logging.getLogger(__name__)


class RAGIndexer:
    """Orchestrates building and updating RAG indexes."""

    def __init__(
        self,
        project_dir: Path,
        index_dir: Path | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.index_dir = index_dir or (self.project_dir / ".rag_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or RAGConfig()
        self._chunker = Chunker(self.config)
        self._checksums_path = self.index_dir / "checksums.json"
        self._code_store = CodeStore(self.index_dir, self.config)
        self._summary_store = SummaryStore(self.index_dir, self.config)

    def _load_checksums(self) -> dict[str, str]:
        if self._checksums_path.exists():
            return json.loads(self._checksums_path.read_text(encoding="utf-8"))
        return {}

    def _save_checksums(self, checksums: dict[str, str]) -> None:
        self._checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    @staticmethod
    def _md5(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    def build(self) -> dict[str, int]:
        """Full build: chunk all files and index summaries from ProjectGraph."""
        logger.info("RAG full build started for %s", self.project_dir)
        logger.info("Index directory: %s", self.index_dir)
        logger.info("Config: chunk_size=%d, overlap=%d, code_model=%s, summary_model=%s",
                     self.config.chunk_size, self.config.chunk_overlap,
                     self.config.code_embedding_model, self.config.summary_embedding_model)

        logger.info("Clearing existing stores...")
        self._code_store.clear()
        self._summary_store.clear()

        logger.info("Collecting files (exclude: %s)...", settings.indexer_exclude_patterns)
        files = self._chunker.collect_files(self.project_dir, settings.indexer_exclude_patterns)
        logger.info("Found %d files to index", len(files))
        checksums: dict[str, str] = {}

        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict] = []

        for i, f in enumerate(files, 1):
            rel = str(f.relative_to(self.project_dir))
            checksums[rel] = self._md5(f)
            chunks = self._chunker.chunk_file(f, self.project_dir)
            if i % 50 == 0 or i == len(files):
                logger.info("Chunking progress: %d/%d files (%d chunks so far)", i, len(files), len(all_ids))
            for chunk in chunks:
                chunk_id = f"{chunk.file_path}::{chunk.chunk_index}"
                all_ids.append(chunk_id)
                all_docs.append(chunk.content)
                all_metas.append({
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                })
            logger.debug("  %s -> %d chunks", rel, len(chunks))

        if all_ids:
            logger.info("Embedding and storing %d code chunks...", len(all_ids))
            self._code_store.add(all_ids, all_docs, all_metas)
            logger.info("Code chunks stored successfully")
        else:
            logger.warning("No code chunks produced from %d files", len(files))

        # Index summaries from ProjectGraph if available
        logger.info("Indexing summaries from ProjectGraph...")
        summary_count = self._index_summaries_from_graph()

        self._save_checksums(checksums)
        logger.info("Checksums saved (%d entries)", len(checksums))

        stats = {
            "files": len(files),
            "chunks": len(all_ids),
            "summaries": summary_count,
        }
        logger.info("RAG build complete: %d files, %d chunks, %d summaries",
                     stats["files"], stats["chunks"], stats["summaries"])
        return stats

    def _stores_populated(self) -> bool:
        """Check whether ChromaDB collections actually contain data."""
        code_count = self._code_store.count()
        summary_count = self._summary_store.count()
        logger.info("Store status: code_chunks=%d, summaries=%d", code_count, summary_count)
        return code_count > 0

    def update(self) -> dict[str, int]:
        """Incremental update based on file checksums.

        If checksums exist but the ChromaDB stores are empty (e.g. DB was
        deleted or never populated), falls back to a full build automatically.
        """
        logger.info("RAG incremental update started for %s", self.project_dir)
        old_checksums = self._load_checksums()
        logger.info("Loaded %d existing checksums", len(old_checksums))

        # Verify that the stores actually have data — checksums.json can
        # survive even if the ChromaDB files were removed.
        if old_checksums and not self._stores_populated():
            logger.warning(
                "Checksums file exists (%d entries) but ChromaDB stores are empty — "
                "falling back to full build",
                len(old_checksums),
            )
            return self.build()

        logger.info("Scanning current files...")
        files = self._chunker.collect_files(self.project_dir, settings.indexer_exclude_patterns)
        new_checksums: dict[str, str] = {}
        for f in files:
            rel = str(f.relative_to(self.project_dir))
            new_checksums[rel] = self._md5(f)
        logger.info("Found %d current files", len(new_checksums))

        # No previous checksums at all — first run, do full build
        if not old_checksums:
            logger.info("No previous checksums found — performing full build")
            return self.build()

        old_files = set(old_checksums.keys())
        new_files = set(new_checksums.keys())

        deleted = old_files - new_files
        added = new_files - old_files
        changed = {f for f in old_files & new_files if old_checksums[f] != new_checksums[f]}
        unchanged = len(old_files & new_files) - len(changed)

        logger.info("File diff: +%d added, ~%d changed, -%d deleted, %d unchanged",
                     len(added), len(changed), len(deleted), unchanged)

        if deleted:
            logger.debug("Deleted files: %s", sorted(deleted))
        if changed:
            logger.debug("Changed files: %s", sorted(changed))
        if added:
            logger.debug("Added files: %s", sorted(added))

        to_remove = deleted | changed
        to_add = added | changed

        # Remove stale chunks
        if to_remove:
            logger.info("Removing stale chunks for %d files...", len(to_remove))
            self._code_store.delete_by_file(list(to_remove))
            self._summary_store.delete_by_file(list(to_remove))
            logger.info("Stale chunks removed")
        else:
            logger.info("No stale chunks to remove")

        # Add new/changed chunks
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict] = []

        if to_add:
            logger.info("Chunking %d new/changed files...", len(to_add))
        for i, rel in enumerate(sorted(to_add), 1):
            f = self.project_dir / rel
            if not f.exists():
                logger.debug("Skipping missing file: %s", rel)
                continue
            chunks = self._chunker.chunk_file(f, self.project_dir)
            logger.debug("  %s -> %d chunks", rel, len(chunks))
            for chunk in chunks:
                chunk_id = f"{chunk.file_path}::{chunk.chunk_index}"
                all_ids.append(chunk_id)
                all_docs.append(chunk.content)
                all_metas.append({
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                })

        if all_ids:
            logger.info("Embedding and storing %d new chunks...", len(all_ids))
            self._code_store.add(all_ids, all_docs, all_metas)
            logger.info("New chunks stored successfully")
        elif to_add:
            logger.warning("No chunks produced from %d files", len(to_add))
        else:
            logger.info("No new chunks to add")

        # Re-index summaries
        logger.info("Re-indexing summaries from ProjectGraph...")
        summary_count = self._index_summaries_from_graph()

        self._save_checksums(new_checksums)
        logger.info("Checksums saved (%d entries)", len(new_checksums))

        stats = {
            "deleted": len(deleted),
            "changed": len(changed),
            "added": len(added),
            "new_chunks": len(all_ids),
            "summaries": summary_count,
        }
        logger.info("RAG update complete: +%d added, ~%d changed, -%d deleted, %d new chunks, %d summaries",
                     stats["added"], stats["changed"], stats["deleted"],
                     stats["new_chunks"], stats["summaries"])
        return stats

    def _index_summaries_from_graph(self) -> int:
        """Load summaries from ProjectGraph and add to summary store."""
        # Look for graph.json in sibling .code_index directory
        code_index = self.project_dir / ".code_index"
        if not code_index.exists():
            # Also check parent-level .code_index (RepoManager layout)
            parent_index = self.index_dir.parent / ".code_index"
            if parent_index.exists():
                code_index = parent_index
                logger.info("Using code index at %s", code_index)
            else:
                logger.info("No code index found at %s or %s, skipping summary indexing",
                            self.project_dir / ".code_index", parent_index)
                return 0
        else:
            logger.info("Using code index at %s", code_index)

        try:
            from coding_agent.code_indexer.graph.storage import load_graph

            graph = load_graph(code_index)
            logger.info("Loaded ProjectGraph: %d nodes, %d edges", len(graph.nodes), len(graph.edges))
        except Exception as e:
            logger.warning("Cannot load project graph from %s: %s", code_index, e)
            return 0

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        seen_ids: set[str] = set()

        for node in graph.nodes:
            if not node.summary:
                continue
            if node.id in seen_ids:
                logger.debug("Skipping duplicate node ID: %s", node.id)
                continue
            seen_ids.add(node.id)
            ids.append(node.id)
            docs.append(node.summary)
            metas.append({
                "node_id": node.id,
                "file_path": node.file_path,
                "node_type": node.node_type.value,
                "name": node.name,
            })

        total_nodes = len(graph.nodes)
        skipped = total_nodes - len(ids)
        logger.info("Summary indexing: %d nodes with summaries, %d without (skipped)", len(ids), skipped)

        if ids:
            logger.info("Clearing and rebuilding summary store...")
            self._summary_store.clear()
            logger.info("Embedding %d summaries with model %s...",
                        len(ids), self.config.summary_embedding_model)
            self._summary_store.add(ids, docs, metas)
            logger.info("Summary store updated: %d entries", len(ids))
        else:
            logger.info("No summaries to index")

        return len(ids)
