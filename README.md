# Coding Agent

Baseline coding agent that automatically solves GitHub issues, reviews PRs, and iterates on feedback.

## Setup

```bash
pip install -e .
cp .env.example .env
# Edit .env with your tokens
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `GITHUB_TOKEN` | GitHub personal access token | — |
| `LLM_API_KEY` | LLM API key (OpenRouter/OpenAI) | — |
| `LLM_BASE_URL` | LLM API base URL | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Model identifier | `openai/gpt-4o-mini` |
| `MAX_CONTEXT_FILES` | Max .py files for context | `10` |
| `MAX_FILE_SIZE_KB` | Max file size to include | `50` |
| `MAX_ITERATIONS` | Max review iterations | `3` |

## Usage

```bash
# Solve an issue
coding-agent solve --repo owner/repo --issue 1

# Review a PR
coding-agent review --repo owner/repo --pr 1

# Iterate on review feedback
coding-agent iterate --repo owner/repo --pr 1
```

## GitHub Actions

Three workflows are included:

- **solve-issue.yml** — Triggered when an issue is labeled `ai-solve`. Creates a PR with a solution.
- **review-pr.yml** — Triggered when a PR labeled `ai-generated` is opened/updated. Posts a review comment.
- **iterate-pr.yml** — Triggered when a comment containing "Changes Requested" is posted on a PR. Pushes fixes.

### Required Secrets

Set these in your repository settings:

- `LLM_API_KEY`
- `LLM_BASE_URL` (optional)
- `LLM_MODEL` (optional)

`GITHUB_TOKEN` is provided automatically by GitHub Actions.

## Architecture

```
CLI (typer) → Agents → Services (GitHub API + LLM)
                ↕
            Prompts (XML-based)
```

- **CodeAgent** — Reads issue + repo context → LLM generates solution → commits to branch → creates PR
- **ReviewerAgent** — Reads PR diff + CI status → LLM reviews → posts comment
- **Iterator** — Reads review feedback → LLM produces fixes → commits to same branch

## Docker

```bash
docker build -t coding-agent .
docker-compose up
```
