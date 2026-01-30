"""Code index query tools wrapping code_indexer/tools.py."""

from __future__ import annotations

from coding_agent.code_indexer import tools as idx_tools
from coding_agent.tools import Tool

NODE_ID_DESC = (
    "Node ID in the format 'relative/path.py::Name'. "
    "Examples: 'src/app.py::main' (function), "
    "'src/models.py::User' (class), "
    "'src/models.py::User.save' (method), "
    "'src/app.py' (module). "
    "Use list_modules or search_entity to discover valid IDs first."
)


def create_indexer_tools(index_dir: str) -> list[Tool]:
    return [
        Tool(
            name="explore_structure",
            description=(
                "Explore project directory structure as a tree with summaries. "
                "Use this first to understand the project layout before diving into files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path to explore (e.g. 'src/coding_agent'). "
                            "Omit to start from root."
                        ),
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many levels deep to show (default 2).",
                        "default": 2,
                    },
                },
                "required": [],
            },
            execute=lambda args: idx_tools.explore_structure(
                path=args.get("path"),
                depth=args.get("depth", 2),
                index_dir=index_dir,
            ),
        ),
        Tool(
            name="list_modules",
            description=(
                "List all indexed modules (files) in the project. "
                "Returns module paths and summaries. "
                "Use this first to discover what files exist."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            execute=lambda args: idx_tools.list_modules(index_dir),
        ),
        Tool(
            name="get_file_structure",
            description=(
                "Show all classes, functions, and methods defined in a file. "
                "Returns node IDs you can pass to other tools. "
                "Example file_path: 'src/services/llm_service.py'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Relative file path as shown by list_modules. "
                            "Example: 'src/coding_agent/cli.py'"
                        ),
                    },
                },
                "required": ["file_path"],
            },
            execute=lambda args: idx_tools.get_file_structure(args["file_path"], index_dir),
        ),
        Tool(
            name="get_node_info",
            description=(
                "Get full context for a code element: signature, summary, "
                "parent, children, dependencies, and dependents."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": NODE_ID_DESC},
                },
                "required": ["node_id"],
            },
            execute=lambda args: idx_tools.get_element_context(args["node_id"], index_dir),
        ),
        Tool(
            name="get_node_connections",
            description=(
                "Get all elements related to a node — imports, calls, "
                "inheritance, containment — with edge type labels."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": NODE_ID_DESC},
                },
                "required": ["node_id"],
            },
            execute=lambda args: idx_tools.get_related_elements(args["node_id"], index_dir),
        ),
        Tool(
            name="search_entity",
            description=(
                "Search for classes, functions, and files by name, docstring, or summary. "
                "Returns matching node IDs with types. "
                "Example queries: 'solve', 'LLMService', 'parse solution'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or phrase. Example: 'generate'",
                    },
                },
                "required": ["query"],
            },
            execute=lambda args: idx_tools.search_entity(args["query"], index_dir),
        ),
        Tool(
            name="get_source_code",
            description=(
                "Get the source code of a specific function, class, or method by node ID. "
                "Use get_file_structure or search_entity to find valid node IDs first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": NODE_ID_DESC},
                },
                "required": ["node_id"],
            },
            execute=lambda args: idx_tools.get_code(args["node_id"], index_dir),
        ),
    ]
