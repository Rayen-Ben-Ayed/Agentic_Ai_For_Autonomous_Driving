import asyncio
import json
import logging
import re
from typing import Any, Optional

from agent.prompt_templates import get_decision_prompt, get_system_prompt
from mcp_interface.client import MCPDrivingClient
from pipeline_log import log_stage
from simulation.timing_config import MAX_LLM_TOOL_ROUNDS

logger = logging.getLogger(__name__)

VALID_ACTIONS = frozenset({
    "overtake",
    "follow_lane",
    "stop",
    "yield",
    "change_lane_left",
    "change_lane_right",
    "go_straight",
    "turn_right",
    "turn_left",
})

# After this many rejected actions in one step, stop deliberating and force a
# safe stop instead of burning more (slow) LLM rounds on the same dead end.
MAX_REJECTIONS_BEFORE_STOP = 2


def _extract_action(content: str | None) -> Optional[str]:
    """Best-effort recovery of the intended action from a plain-text reply.

    Function-calling models sometimes emit the final decision as content (e.g.
    ``{"action": "follow_lane"}``) instead of an ``execute_action`` tool call.
    Prefer an explicit JSON ``action`` field; fall back to a whole-word scan.
    Returns a valid action name or None.
    """
    if not content:
        return None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            action = parsed.get("action")
            if isinstance(action, str) and action in VALID_ACTIONS:
                return action
    except (json.JSONDecodeError, TypeError):
        pass
    for action in VALID_ACTIONS:
        if re.search(rf"\b{re.escape(action)}\b", content):
            return action
    return None


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
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": get_decision_prompt()},
        ]

        executed_action: Optional[str] = None
        rejection_count = 0

        for round_idx in range(1, MAX_LLM_TOOL_ROUNDS + 1):
            log_stage(
                logger, "agent", "LLM round %d/%d", round_idx, MAX_LLM_TOOL_ROUNDS
            )
            response_message = self.llm_client.generate_response(messages, tools=tools)
            if not response_message:
                return await self._force_safe_stop("LLM returned no response")

            if not response_message.tool_calls:
                logger.info("LLM response without tool call: %s", response_message.content)
                # The model stated its decision as text instead of calling
                # execute_action. Try to honor it (still gated by MCP preview /
                # allowed_actions) rather than slamming a hard stop on a clear
                # road; only fall back to stop when nothing valid can be applied.
                recovered = _extract_action(response_message.content)
                if recovered:
                    result_text = await self.mcp_client.call_tool(
                        "execute_action", {"action": recovered}
                    )
                    try:
                        result = json.loads(result_text)
                    except json.JSONDecodeError:
                        result = {}
                    if result.get("status") == "success":
                        log_stage(
                            logger, "agent", "recovered text action -> %s", recovered
                        )
                        return recovered
                if round_idx < MAX_LLM_TOOL_ROUNDS:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You did not call a tool. Call execute_action with "
                                "your chosen action (after preview_action). Do not "
                                "reply in plain text."
                            ),
                        }
                    )
                    continue
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
