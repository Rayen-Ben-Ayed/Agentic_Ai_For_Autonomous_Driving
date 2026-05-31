import json
import logging

from agent.prompt_templates import SYSTEM_PROMPT
from simulation.agent_tools import execute_action, get_world_state

logger = logging.getLogger(__name__)


def _parse_tool_arguments(raw) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


class DecisionMaker:
    def __init__(self, llm_client, mcp_server=None, system_prompt: str | None = None):
        self.llm_client = llm_client
        self.mcp_server = mcp_server
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.messages = [{"role": "system", "content": self.system_prompt}]

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_world_state",
                    "description": "Retrieves the current world state from the Phabmacs simulation in JSON format.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_action",
                    "description": "Executes a discrete driving action in the Phabmacs simulation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "overtake",
                                    "follow_lane",
                                    "stop",
                                    "yield",
                                    "change_lane_left",
                                    "change_lane_right",
                                ],
                                "description": "The action to execute.",
                            }
                        },
                        "required": ["action"],
                    },
                },
            },
        ]

    def make_decision(self, user_message: str | None = None):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.messages.append({
            "role": "user",
            "content": user_message or "Analyze the current world state and execute the next action.",
        })

        max_iterations = 5
        for _ in range(max_iterations):
            response_message = self.llm_client.generate_response(self.messages, tools=self.tools)

            if not response_message:
                logger.error("No response from LLM.")
                break

            self.messages.append(response_message)

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = _parse_tool_arguments(tool_call.function.arguments)

                    logger.info("LLM called tool: %s with args: %s", function_name, function_args)

                    if function_name == "get_world_state":
                        result = get_world_state()
                    elif function_name == "execute_action":
                        result = execute_action(function_args.get("action", ""))
                    else:
                        result = json.dumps({"error": f"Unknown tool: {function_name}"})

                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result,
                    })

                    if function_name == "execute_action":
                        return function_args.get("action")
            else:
                logger.info("LLM response (no tool call): %s", response_message.content)
                break

        return None
