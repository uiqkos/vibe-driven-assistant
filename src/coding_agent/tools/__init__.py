"""Tool plugin system for the agentic loop."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list_all(self) -> list[Tool]:
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        result = []
        for tool in self._tools.values():
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return result

    def execute(self, name: str, args: dict[str, Any]) -> str:
        try:
            tool = self._tools[name]
            return tool.execute(args)
        except KeyError:
            return f"Error: unknown tool '{name}'"
        except Exception as e:
            logger.error("Tool '%s' failed: %s", name, e)
            return f"Error executing '{name}': {e}\n{traceback.format_exc()}"
