import json
import logging
from agent.prompt_templates import SYSTEM_PROMPT, get_decision_prompt

logger = logging.getLogger(__name__)

class DecisionMaker:
    def __init__(self, llm_client, mcp_server):
        self.llm_client = llm_client
        self.mcp_server = mcp_server
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Define tools for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_world_state",
                    "description": "Retrieves the current world state from the CARLA simulation in JSON format.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_action",
                    "description": "Executes a discrete driving action in the CARLA simulation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["overtake", "follow_lane", "stop", "yield", "change_lane_left", "change_lane_right"],
                                "description": "The action to execute."
                            }
                        },
                        "required": ["action"]
                    }
                }
            }
        ]

    def make_decision(self):
        """
        Runs the decision loop for a single step.
        """
        # Reset messages for each decision to avoid hallucinating based on long history
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages.append({"role": "user", "content": "Analyze the current world state and execute the next action."})

        # Loop to handle tool calls
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
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"LLM called tool: {function_name} with args: {function_args}")
                    
                    # Execute the tool via the MCP server (simulated direct call here for skeleton)
                    # In a full MCP setup, this would be an MCP client request
                    if function_name == "get_world_state":
                        # We call the underlying function registered in FastMCP
                        # For the skeleton, we can import the function directly or use the FastMCP instance
                        from mcp_interface.server import get_world_state
                        result = get_world_state()
                    elif function_name == "execute_action":
                        from mcp_interface.server import execute_action
                        result = execute_action(function_args.get("action"))
                    else:
                        result = json.dumps({"error": f"Unknown tool: {function_name}"})

                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result,
                    })
                    
                    # If action was executed, we can consider the decision step complete
                    if function_name == "execute_action":
                        return function_args.get("action")
            else:
                # No tool calls, LLM just responded with text
                logger.info(f"LLM response: {response_message.content}")
                break
                
        return None
