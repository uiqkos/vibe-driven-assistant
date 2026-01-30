# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
pip install -e .                # dev install
pytest                          # run all tests
pytest tests/test_basic.py      # single test file
ruff check src tests            # lint
ruff format src tests           # format
mypy src                        # type check
coding-agent solve --repo owner/repo --issue 1
coding-agent review --repo owner/repo --pr 1
coding-agent iterate --repo owner/repo --pr 1
coding-agent local-solve --task "..." -f file1.py     # solve without GitHub
coding-agent agentic-solve --task "..." --local -f file1.py  # agentic tool-calling loop
coding-agent index build --project . --no-llm         # build code index (no LLM)
coding-agent index search --query "LLM"               # search index
coding-agent serve --port 8000                        # webhook server (GitHub App mode)
```

Docker: `docker build -t coding-agent . && docker-compose up`

## Configuration

Pydantic Settings from `.env` (see `.env.example`). Key vars: `GITHUB_TOKEN`, `LLM_API_KEY`, `LLM_BASE_URL` (default: OpenRouter), `LLM_MODEL` (default: gpt-4o-mini). Optional `models.yaml` provides per-agent model overrides (solve, review, iterate, agentic, indexer) — see `models.yaml.example`.

## Architecture

```
CLI (Typer) → Agents → Services (GitHub + LLM)
                ↓
             Prompts (XML templates)
```

**Three agents** share a common pattern — fetch context, build XML prompt, call LLM, parse XML response, act:
- `CodeAgent.solve_issue()` — reads issue + repo files → LLM generates `<solution>` with `<file>` blocks → commits to `issue-N` branch → creates PR
- `ReviewerAgent.review_pr()` — reads diff + issue + CI → LLM generates `<review>` with status/issues → posts comment
- `Iterator.iterate()` — reads review feedback → LLM generates fixes (max 3 cycles) → commits to same branch
- `AgenticAgent` — multi-step tool-calling agent (up to 30 steps) with `ToolRegistry`; can use code indexer tools for navigation

**Services**: `GitHubService` and `LocalService` both implement the `FileService` abstract interface — use `LocalService` for non-GitHub workflows. `LLMService` wraps OpenAI client (OpenRouter-compatible) with retry (3 attempts, exponential backoff, temp=0.2). `RepoManager` handles git operations (clone, branch, commit, push).

**Tool System**: `ToolRegistry` in `tools/` dynamically registers tool functions and converts them to OpenAI function-calling format. Base tools: `view_file`, `edit_file`, `create_file`, `list_directory`. Indexer tools wrap the code indexer queries.

**Prompts** in `prompts/` return XML-structured strings. Agents parse responses with regex to extract file changes or review results.

**Key Models** (`models/`): `CodeSolution` (file changes + explanation), `ReviewResult` (status/issues/CI), `AgentResult`. Code indexer models: `CodeNode`, `Edge` (with types IMPORTS/CALLS/INHERITS/CONTAINS), `ProjectGraph`.

**Code Indexer** (`code_indexer/`) — standalone subsystem that builds a navigation graph of Python repos:
- Pipeline: scan .py files → AST parse (with LLM fallback) → resolve cross-file edges → generate LLM summaries → store as JSON
- Node ID format: `relative/path.py::Class.method`
- Incremental updates via MD5 checksums per file
- `tools.py` exposes 6 query functions for agent consumption (list_modules, search_code, get_file_structure, get_code, get_element_context, get_related_elements)

## GitHub Actions Workflows

Three workflows in `.github/workflows/` trigger on labels/comments:
- `solve-issue.yml` — on `ai-solve` label
- `review-pr.yml` — on `ai-generated` label
- `iterate-pr.yml` — on "Changes Requested" comment
