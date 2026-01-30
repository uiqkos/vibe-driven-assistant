#!/usr/bin/env python3
"""Backfill checksums and git_head in an existing graph.json without re-parsing."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/backfill_graph.py <index_dir>")
        print("  index_dir — directory with graph.json (e.g. /path/to/repo/.code_index)")
        sys.exit(1)

    index_dir = Path(sys.argv[1]).resolve()
    graph_path = index_dir / "graph.json"

    if not graph_path.exists():
        print(f"Error: {graph_path} not found")
        sys.exit(1)

    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    # Use root_path from the graph itself — this is the actual source root
    root_path = Path(graph.get("root_path", ""))
    if not root_path.is_dir():
        print(f"Error: root_path from graph ({root_path}) is not a valid directory")
        sys.exit(1)
    print(f"root_path = {root_path}")

    # --- git_head ---
    # Try root_path first, then walk up to find .git
    git_dir = root_path
    for _ in range(5):
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=git_dir, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            break
        except Exception:
            git_dir = git_dir.parent
    else:
        head = ""
    graph["git_head"] = head
    print(f"git_head = {head[:8] if head else '(not a git repo)'}")

    # --- checksums ---
    node_files = {n.get("file_path", "") for n in graph.get("nodes", [])}
    node_files.discard("")

    checksums = []
    missing = []
    for rel in sorted(node_files):
        abs_path = root_path / rel
        if abs_path.is_file():
            md5 = hashlib.md5(abs_path.read_bytes()).hexdigest()
            checksums.append({"file_path": rel, "md5": md5})
        else:
            missing.append(rel)

    graph["checksums"] = checksums
    print(f"checksums: {len(checksums)} files")
    if missing:
        print(f"missing:   {len(missing)} files (not found on disk)")
        for m in missing[:5]:
            print(f"  - {m}")

    # --- save ---
    graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {graph_path}")


if __name__ == "__main__":
    main()
