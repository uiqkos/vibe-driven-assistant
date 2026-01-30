"""Rich console callback for the agentic loop."""

from __future__ import annotations

from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from coding_agent.tools import ToolRegistry

GITHUB_BODY_LIMIT = 65536

MAX_RESULT_LINES = 30
MAX_RESULT_CHARS = 2000


def _truncate(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > MAX_RESULT_LINES or len(text) > MAX_RESULT_CHARS:
        kept = lines[:MAX_RESULT_LINES]
        truncated = "\n".join(kept)
        if len(truncated) > MAX_RESULT_CHARS:
            truncated = truncated[:MAX_RESULT_CHARS]
        omitted = len(lines) - MAX_RESULT_LINES
        if omitted > 0:
            truncated += f"\n... ({omitted} more lines)"
        return truncated
    return text


TOOL_ICONS = {
    "view_file": "👁 ",
    "edit_file": "✏️ ",
    "create_file": "📄",
    "list_directory": "📂",
    "search_entity": "🔍",
    "list_modules": "📦",
    "get_file_structure": "🏗 ",
    "get_node_info": "🔎",
    "get_node_connections": "🔗",
    "get_source_code": "📝",
}


def _format_arg_value(value: Any) -> str:
    """Format a single argument value — truncate long strings."""
    s = str(value)
    if len(s) > 120:
        return s[:120] + "..."
    return s


class ConsoleCallback:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_tools(self, registry: ToolRegistry) -> None:
        table = Table(title="Available tools", border_style="dim", show_lines=False)
        table.add_column("Tool", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="dim")
        for tool in registry.list_all():
            icon = TOOL_ICONS.get(tool.name, "🔧")
            params = tool.parameters.get("properties", {})
            param_names = ", ".join(params.keys()) if params else ""
            name_col = f"{icon} {tool.name}({param_names})"
            table.add_row(name_col, tool.description)
        self.console.print(table)
        self.console.print()

    def on_step_start(self, step: int, max_steps: int) -> None:
        self.console.rule(f"[bold blue]Step {step}/{max_steps}", style="blue")

    def on_thinking(self, text: str) -> None:
        self.console.print(
            Panel(
                _truncate(text),
                title="[bold yellow]Thinking",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        icon = TOOL_ICONS.get(name, "🔧")
        self.console.print(f"  {icon} [bold cyan]{name}[/]")
        if args:
            for k, v in args.items():
                val = _format_arg_value(v)
                # Multiline values (e.g. old_text / new_text / content) get a panel
                if "\n" in val:
                    self.console.print(f"      [dim]{k}:[/]")
                    self.console.print(
                        Panel(
                            Syntax(val, "python", theme="ansi_dark", word_wrap=True),
                            border_style="dim",
                            padding=(0, 1),
                        )
                    )
                else:
                    self.console.print(f"      [dim]{k}:[/] {val}")

    def on_tool_result(self, name: str, result: str) -> None:
        truncated = _truncate(result)
        self.console.print(
            Panel(
                Syntax(truncated, "text", theme="ansi_dark", word_wrap=True)
                if len(truncated) > 200
                else Text(truncated, style="dim"),
                title="[dim]result",
                border_style="dim",
                padding=(0, 1),
            )
        )

    def on_finish(self, text: str, steps: int, tool_calls: int) -> None:
        self.console.print()
        self.console.rule("[bold green]Agent finished", style="green")
        self.console.print(
            Panel(
                text,
                title=f"[bold green]Result ({steps} steps, {tool_calls} tool calls)",
                border_style="green",
                padding=(0, 1),
            )
        )


class MarkdownCallback:
    """Collects agent events into markdown for use in PR bodies."""

    def __init__(self) -> None:
        self._entries: list[str] = []
        self._current_step: int = 0
        self._finish_text: str = ""
        self._total_steps: int = 0
        self._total_tool_calls: int = 0

    def on_step_start(self, step: int, max_steps: int) -> None:
        self._current_step = step

    def on_thinking(self, text: str) -> None:
        block = (
            f"<details><summary>Step {self._current_step} — 💭 Thinking</summary>\n\n"
            f"{text}\n\n"
            f"</details>"
        )
        self._entries.append(block)

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        icon = TOOL_ICONS.get(name, "🔧")
        short_args = ", ".join(
            f'{k}="{_format_arg_value(v)}"' for k, v in args.items()
        )
        header = f"Step {self._current_step} — {icon} {name}({short_args})"
        # Result will be appended by on_tool_result
        self._entries.append(
            f"<details><summary>{header}</summary>\n\n"
        )

    # Tools whose output should be rendered as plain text (not in a code block)
    _PLAIN_OUTPUT_TOOLS = {"get_file_structure", "search_entity"}
    # Tools whose output may contain backtick fences that break markdown nesting
    _STRIP_FENCES_TOOLS = {"get_source_code"}

    def on_tool_result(self, name: str, result: str) -> None:
        truncated = _truncate(result)
        if name in self._STRIP_FENCES_TOOLS:
            truncated = truncated.replace("```", "")
        if self._entries:
            if name in self._PLAIN_OUTPUT_TOOLS:
                self._entries[-1] += f"{truncated}\n\n</details>"
            else:
                self._entries[-1] += f"```\n{truncated}\n```\n\n</details>"

    def on_finish(self, text: str, steps: int, tool_calls: int) -> None:
        self._finish_text = text
        self._total_steps = steps
        self._total_tool_calls = tool_calls

    def build_pr_body(self, issue_number: int, issue_title: str) -> str:
        """Build a GitHub PR body with collapsible agent log."""
        parts: list[str] = [
            f"Resolves #{issue_number}",
            "",
            "**Agent summary:**",
            self._finish_text or "_No final message._",
            "",
        ]

        if self._entries:
            inner = "\n\n".join(self._entries)
            summary = (
                f"Agent log ({self._total_steps} steps, "
                f"{self._total_tool_calls} tool calls)"
            )
            parts.append(
                f"<details><summary>{summary}</summary>\n\n"
                f"{inner}\n\n"
                f"</details>"
            )

        body = "\n".join(parts)
        if len(body) > GITHUB_BODY_LIMIT:
            budget = GITHUB_BODY_LIMIT - 200
            body = body[:budget] + "\n\n_(log truncated)_\n\n</details>"
        return body


class CompositeCallback:
    """Forwards all callback events to multiple delegates."""

    def __init__(self, callbacks: Sequence[Any]) -> None:
        self._callbacks = callbacks

    def on_step_start(self, step: int, max_steps: int) -> None:
        for cb in self._callbacks:
            cb.on_step_start(step, max_steps)

    def on_thinking(self, text: str) -> None:
        for cb in self._callbacks:
            cb.on_thinking(text)

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_tool_call(name, args)

    def on_tool_result(self, name: str, result: str) -> None:
        for cb in self._callbacks:
            cb.on_tool_result(name, result)

    def on_finish(self, text: str, steps: int, tool_calls: int) -> None:
        for cb in self._callbacks:
            cb.on_finish(text, steps, tool_calls)
