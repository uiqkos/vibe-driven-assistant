"""Cross-file resolution of import/call edges to actual node IDs."""

from __future__ import annotations

from pathlib import PurePosixPath

from coding_agent.code_indexer.models import CodeNode, Edge, EdgeType


# Extension to language mapping for import resolution
_EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


def _detect_language(file_path: str) -> str:
    """Determine language from file path extension."""
    suffix = PurePosixPath(file_path).suffix
    return _EXT_LANGUAGE.get(suffix, "python")


def resolve_edges(
    nodes: list[CodeNode],
    edges: list[Edge],
) -> list[Edge]:
    """Resolve name-based edge targets to actual node IDs where possible."""
    node_ids = {n.id for n in nodes}
    # Build lookup: short name -> list of node IDs
    name_to_ids: dict[str, list[str]] = {}
    for n in nodes:
        name_to_ids.setdefault(n.name, []).append(n.id)
        # Also index by last segment (e.g. "Class.method" -> method id)
        parts = n.name.rsplit(".", 1)
        if len(parts) == 2:
            name_to_ids.setdefault(parts[-1], []).append(n.id)

    # Module name -> module node id (for import resolution)
    module_to_id: dict[str, str] = {}
    for n in nodes:
        if n.node_type.value == "module":
            lang = _detect_language(n.file_path)
            if lang == "python":
                # src/coding_agent/config.py -> coding_agent.config
                mod_name = n.file_path.replace("/", ".").removesuffix(".py")
                if mod_name.startswith("src."):
                    mod_name = mod_name[4:]
                module_to_id[mod_name] = n.id
            elif lang in ("javascript", "typescript"):
                # JS/TS: relative imports like ./utils or ../helpers
                # Store by relative path without extension
                p = PurePosixPath(n.file_path)
                for ext in (".js", ".jsx", ".ts", ".tsx"):
                    if n.file_path.endswith(ext):
                        module_to_id[str(p.with_suffix(""))] = n.id
                        break
                # Also store with extension
                module_to_id[n.file_path] = n.id
            elif lang == "go":
                # Go: imports are full package paths; store by directory (package)
                p = PurePosixPath(n.file_path)
                module_to_id[str(p.parent)] = n.id
                module_to_id[n.file_path] = n.id
            elif lang == "rust":
                # Rust: use crate::module::item; store by crate-relative path
                p = PurePosixPath(n.file_path)
                # src/lib.rs, src/main.rs, src/foo.rs -> crate::foo
                no_ext = str(p.with_suffix(""))
                parts_list = no_ext.split("/")
                if parts_list and parts_list[0] == "src":
                    parts_list = parts_list[1:]
                # Remove lib/main as they map to crate root
                if parts_list and parts_list[-1] in ("lib", "main", "mod"):
                    parts_list = parts_list[:-1]
                crate_path = "::".join(["crate"] + parts_list) if parts_list else "crate"
                module_to_id[crate_path] = n.id
                module_to_id[n.file_path] = n.id

    resolved: list[Edge] = []
    for e in edges:
        target = e.target
        if target in node_ids:
            resolved.append(e)
            continue

        # Try module resolution for imports
        if e.edge_type == EdgeType.IMPORTS:
            if target in module_to_id:
                resolved.append(Edge(source=e.source, target=module_to_id[target], edge_type=e.edge_type))
                continue

            # JS relative import resolution
            source_file = e.source.split("::")[0]
            lang = _detect_language(source_file)
            if lang in ("javascript", "typescript") and target.startswith("."):
                source_dir = str(PurePosixPath(source_file).parent)
                resolved_path = str(PurePosixPath(source_dir) / target)
                # Normalize (remove ./ and resolve ..)
                resolved_path = str(PurePosixPath(resolved_path))
                if resolved_path in module_to_id:
                    resolved.append(Edge(source=e.source, target=module_to_id[resolved_path], edge_type=e.edge_type))
                    continue
                # Try with extensions
                found = False
                for ext in (".js", ".jsx", ".ts", ".tsx"):
                    candidate = resolved_path + ext
                    if candidate in module_to_id:
                        resolved.append(Edge(source=e.source, target=module_to_id[candidate], edge_type=e.edge_type))
                        found = True
                        break
                    # Try index file
                    idx = resolved_path + "/index" + ext
                    if idx in module_to_id:
                        resolved.append(Edge(source=e.source, target=module_to_id[idx], edge_type=e.edge_type))
                        found = True
                        break
                if found:
                    continue

        # Try name-based resolution
        candidates = name_to_ids.get(target, [])
        if len(candidates) == 1:
            resolved.append(Edge(source=e.source, target=candidates[0], edge_type=e.edge_type))
        elif len(candidates) > 1:
            # Prefer node in same file
            source_file = e.source.split("::")[0]
            same_file = [c for c in candidates if c.startswith(source_file)]
            if same_file:
                resolved.append(Edge(source=e.source, target=same_file[0], edge_type=e.edge_type))
            else:
                resolved.append(Edge(source=e.source, target=candidates[0], edge_type=e.edge_type))
        else:
            # Keep unresolved edge as-is (target stays as name string)
            resolved.append(e)

    return resolved
