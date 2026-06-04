import asyncio
import json
import logging
from typing import Any, Optional

from agent.prompt_templates import SYSTEM_PROMPT, get_decision_prompt
from mcp_interface.client import MCPDrivingClient
from pipeline_log import log_stage

logger = logging.getLogger(__name__)

VALID_ACTIONS = frozenset({
    "overtake",
    "follow_lane",
    "stop",
    "yield",
    "change_lane_left",
    "change_lane_right",
})

MAX_TOOL_ROUNDS = 6

# After this many rejected actions in one step, stop deliberating and force a
# safe stop instead of burning more (slow) LLM rounds on the same dead end.
MAX_REJECTIONS_BEFORE_STOP = 2


class DecisionMaker:
    def __init__(self, llm_client, mcp_client: MCPDrivingClient):
        self.llm_client = llm_client
        self.mcp_client = mcp_client

    def run_step(self) -> Optional[str]:
        """
        One agent step: LLM uses MCP tools to read CARLA state and execute an action.
        execute_action is applied in CARLA via the MCP server when the LLM calls it.
        """
        return asyncio.run(self._run_step_async())

    async def _force_safe_stop(self, reason: str) -> Optional[str]:
        """Deterministic safety net: apply `stop` directly via MCP.

        `stop` is never rejected by the policy, so this guarantees the vehicle
        gets a defensive command instead of coasting on a stale control when the
        agent fails to choose a valid action.
        """
        log_stage(logger, "agent", "fallback -> stop (%s)", reason)
        result_text = await self.mcp_client.call_tool("execute_action", {"action": "stop"})
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            result = {}
        if result.get("status") == "success":
            return "stop"
        logger.error("Safe-stop fallback did not apply: %s", result_text)
        return None

    async def _run_step_async(self) -> Optional[str]:
        tools = await self.mcp_client.get_openai_tools()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_decision_prompt()},
        ]

        executed_action: Optional[str] = None
        rejection_count = 0

        for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
            log_stage(logger, "agent", "LLM round %d/%d", round_idx, MAX_TOOL_ROUNDS)
            response_message = self.llm_client.generate_response(messages, tools=tools)
            if not response_message:
                return await self._force_safe_stop("LLM returned no response")

            if not response_message.tool_calls:
                logger.info("LLM response without tool call: %s", response_message.content)
                return await self._force_safe_stop("LLM produced no tool call")

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response_message.tool_calls
                ],
            }
            messages.append(assistant_message)

            for tool_call in response_message.tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                result_text = await self.mcp_client.call_tool(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

                if name == "execute_action":
                    action = arguments.get("action")
                    try:
                        result = json.loads(result_text)
                    except json.JSONDecodeError:
                        result = {}
                    if result.get("status") == "success" and action in VALID_ACTIONS:
                        log_stage(logger, "agent", "step complete -> %s", action)
                        executed_action = action
                        break
                    elif result.get("status") == "rejected":
                        rejection_count += 1
                        log_stage(
                            logger,
                            "agent",
                            "MCP rejected %s (%d/%d): %s",
                            action,
                            rejection_count,
                            MAX_REJECTIONS_BEFORE_STOP,
                            result.get("message"),
                        )
                    elif action:
                        log_stage(logger, "agent", "action not applied: %s | %s", action, result)

            if executed_action:
                return executed_action

            if rejection_count >= MAX_REJECTIONS_BEFORE_STOP:
                return await self._force_safe_stop(
                    f"{rejection_count} rejected actions"
                )

        logger.warning("Agent exceeded max MCP tool rounds without execute_action")
        return await self._force_safe_stop("max tool rounds exhausted")
