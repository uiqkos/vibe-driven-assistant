"""ReAct-style agentic loop with tool calling."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from coding_agent.models.agent_schemas import AgentResult
from coding_agent.prompts.prompt_layer import load_prompt
from coding_agent.services.llm_service import LLMService
from coding_agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = load_prompt("agentic_system")

class StepCallback(Protocol):
    def on_step_start(self, step: int, max_steps: int) -> None: ...
    def on_thinking(self, text: str) -> None: ...
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_finish(self, text: str, steps: int, tool_calls: int) -> None: ...


class NullCallback:
    def on_step_start(self, step: int, max_steps: int) -> None: ...
    def on_thinking(self, text: str) -> None: ...
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_finish(self, text: str, steps: int, tool_calls: int) -> None: ...


class AgenticAgent:
    def __init__(
        self,
        llm: LLMService,
        registry: ToolRegistry,
        max_steps: int = 30,
        callback: StepCallback | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.cb: StepCallback = callback or NullCallback()

    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        tools = self.registry.to_openai_tools()
        total_tool_calls = 0

        warn_threshold = 5  # warn when this many steps remain
        last_output = ""

        for step in range(self.max_steps):
            remaining = self.max_steps - step
            self.cb.on_step_start(step + 1, self.max_steps)

            # Inject a deadline warning when steps are running low
            if remaining == warn_threshold:
                messages.append({
                    "role": "user",
                    "content": (
                        f"⚠️ You have {remaining} steps remaining. "
                        "Wrap up your changes now — apply any pending edits, "
                        "verify critical files, and respond with a summary. "
                        "Do NOT start new explorations."
                    ),
                })
                logger.info("Injected step budget warning (%d remaining)", remaining)
            elif remaining == 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "🛑 LAST STEP. You must respond with a final summary now. "
                        "Do not make any more tool calls."
                    ),
                })
                logger.info("Injected final step warning")

            response = self.llm.generate_with_tools(messages, tools)
            choice = response.choices[0]
            message = choice.message

            messages.append(message.model_dump(exclude_none=True))

            # Show thinking text if present alongside tool calls
            if message.content and message.tool_calls:
                self.cb.on_thinking(message.content)

            if message.content:
                last_output = message.content

            if not message.tool_calls:
                self.cb.on_finish(message.content or "", step + 1, total_tool_calls)
                return AgentResult(
                    output=message.content or "",
                    steps=step + 1,
                    tool_calls_made=total_tool_calls,
                )

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                self.cb.on_tool_call(name, args)
                result = self.registry.execute(name, args)
                total_tool_calls += 1
                self.cb.on_tool_result(name, result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        # Graceful finish: return what we have instead of crashing
        logger.warning("Agent hit max steps (%d), returning partial result", self.max_steps)
        self.cb.on_finish(last_output, self.max_steps, total_tool_calls)
        return AgentResult(
            output=last_output or "Agent reached step limit before completing.",
            steps=self.max_steps,
            tool_calls_made=total_tool_calls,
        )
