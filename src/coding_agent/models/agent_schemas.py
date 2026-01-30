"""Models for the agentic loop."""

from __future__ import annotations

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class AgentResult(BaseModel):
    output: str
    steps: int
    tool_calls_made: int


class MaxStepsError(Exception):
    """Raised when the agent exceeds the maximum number of steps."""
