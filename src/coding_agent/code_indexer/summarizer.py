"""Batch LLM summary generation for code nodes."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from coding_agent.code_indexer.models import CodeNode, Edge, EdgeType, NodeType

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a code documentation assistant. For each code element, write a concise 1-2 sentence summary \
describing what it does and its purpose. Return ONLY the summary text, no markdown or formatting."""


def summarize_nodes(
    nodes: list[CodeNode],
    root_path: Path,
    llm_service: object,  # LLMService
    batch_size: int = 10,
) -> None:
    """Generate summaries for nodes that don't have one. Mutates nodes in place."""
    to_summarize = [n for n in nodes if not n.summary and n.node_type != NodeType.MODULE]
    total = len(to_summarize)

    if total == 0:
        logger.info("No nodes require summarization")
        return

    for node in to_summarize:
        logger.info("  Will summarize: %s (%s)", node.id, node.node_type.value)

    logger.info("Starting summarization: %d nodes in batches of %d", total, batch_size)
    start_time = time.monotonic()
    succeeded = 0
    failed = 0

    for i in range(0, total, batch_size):
        batch = to_summarize[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        logger.info("Batch %d/%d (%d nodes)", batch_num, total_batches, len(batch))

        for node in batch:
            try:
                source_snippet = _get_source(node, root_path)
                prompt = (
                    f"Summarize this {node.node_type.value} named '{node.name}':\n\n"
                    f"Signature: {node.signature}\n"
                    f"Docstring: {node.docstring}\n\n"
                    f"```python\n{source_snippet}\n```"
                )
                summary = llm_service.generate(prompt, system_prompt=SYSTEM_PROMPT)  # type: ignore[attr-defined]
                node.summary = summary.strip()
                succeeded += 1
                logger.info("  [%d/%d] Summarized: %s", succeeded + failed, total, node.id)
            except Exception:
                node.summary = node.docstring or ""
                failed += 1
                logger.warning("  [%d/%d] Failed: %s (using docstring fallback)", succeeded + failed, total, node.id, exc_info=True)

        logger.info("Progress: %d/%d nodes done", i + len(batch), total)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Summarization complete: %d succeeded, %d failed, %.1fs elapsed",
        succeeded, failed, elapsed,
    )


MODULE_SYSTEM_PROMPT = """\
You are a code documentation assistant. Given a file's top-level definitions with their summaries, \
write a concise 1-sentence summary describing the file's purpose. Return ONLY the summary text."""


def summarize_modules(
    nodes: list[CodeNode],
    edges: list[Edge],
    llm_service: object,
) -> None:
    """Generate summaries for MODULE nodes based on their children. Mutates in place."""
    node_map = {n.id: n for n in nodes}

    # Build children map: module_id -> [child nodes]
    children_map: dict[str, list[CodeNode]] = {}
    for e in edges:
        if e.edge_type == EdgeType.CONTAINS and not e.source.startswith("dir:"):
            parent = node_map.get(e.source)
            child = node_map.get(e.target)
            if parent and parent.node_type == NodeType.MODULE and child:
                children_map.setdefault(e.source, []).append(child)

    modules = [n for n in nodes if n.node_type == NodeType.MODULE and not n.summary]
    if not modules:
        logger.info("No module nodes require summarization")
        return

    logger.info("Summarizing %d module nodes", len(modules))
    for mod in modules:
        children = children_map.get(mod.id, [])
        if not children:
            continue

        child_descriptions = []
        for child in children:
            kind = child.node_type.value
            summary_part = f" — {child.summary}" if child.summary else ""
            sig_part = f" `{child.signature}`" if child.signature else ""
            child_descriptions.append(f"  [{kind}]{sig_part} {child.name}{summary_part}")

        contents = "\n".join(child_descriptions)
        prompt = (
            f"Summarize this Python file '{mod.name}' based on its definitions:\n\n"
            f"{contents}"
        )
        try:
            summary = llm_service.generate(prompt, system_prompt=MODULE_SYSTEM_PROMPT)  # type: ignore[attr-defined]
            mod.summary = summary.strip()
            logger.info("  Summarized module: %s", mod.id)
        except Exception:
            mod.summary = ""
            logger.warning("  Failed to summarize module: %s", mod.id, exc_info=True)


DIR_SYSTEM_PROMPT = """\
You are a code documentation assistant. Given a directory's contents with their summaries, \
write a concise 1-sentence summary describing the directory's purpose. Return ONLY the summary text."""


def summarize_directories(
    nodes: list[CodeNode],
    edges: list[Edge],
    llm_service: object,
) -> None:
    """Generate summaries for DIRECTORY nodes. Processes bottom-up (deepest first). Mutates in place."""
    dir_nodes = [n for n in nodes if n.node_type == NodeType.DIRECTORY and not n.summary]
    if not dir_nodes:
        logger.info("No directory nodes require summarization")
        return

    # Build a lookup: node_id -> node
    node_map = {n.id: n for n in nodes}

    # Build children map from CONTAINS edges where source is a directory
    children_map: dict[str, list[str]] = {}
    for e in edges:
        if e.edge_type == EdgeType.CONTAINS and e.source.startswith("dir:"):
            children_map.setdefault(e.source, []).append(e.target)

    # Sort deepest first
    dir_nodes.sort(key=lambda n: n.file_path.count("/"), reverse=True)

    logger.info("Summarizing %d directory nodes", len(dir_nodes))
    for dn in dir_nodes:
        child_ids = children_map.get(dn.id, [])
        child_descriptions = []
        for cid in child_ids:
            child = node_map.get(cid)
            if child:
                kind = "dir" if child.node_type == NodeType.DIRECTORY else "file"
                summary_part = f" — {child.summary}" if child.summary else ""
                child_descriptions.append(f"  [{kind}] {child.name}{summary_part}")

        if not child_descriptions:
            dn.summary = ""
            continue

        contents = "\n".join(child_descriptions)
        prompt = (
            f"Summarize this directory '{dn.name}' based on its contents:\n\n"
            f"{contents}"
        )
        try:
            summary = llm_service.generate(prompt, system_prompt=DIR_SYSTEM_PROMPT)  # type: ignore[attr-defined]
            dn.summary = summary.strip()
            logger.info("  Summarized directory: %s", dn.id)
        except Exception:
            dn.summary = ""
            logger.warning("  Failed to summarize directory: %s", dn.id, exc_info=True)


def _get_source(node: CodeNode, root_path: Path) -> str:
    """Extract source lines for a node."""
    try:
        path = root_path / node.file_path
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        snippet = lines[node.line_start - 1 : node.line_end]
        # Limit to 50 lines for LLM context
        if len(snippet) > 50:
            snippet = snippet[:50] + ["    # ... (truncated)"]
        return "\n".join(snippet)
    except (OSError, IndexError):
        return ""
