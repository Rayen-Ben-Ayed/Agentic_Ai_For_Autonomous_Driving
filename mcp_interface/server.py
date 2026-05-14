from mcp.server.fastmcp import FastMCP
import json
import logging

# We will inject the simulation components into the MCP server
# For a real implementation, we might use global state or a singleton,
# but FastMCP allows us to define tools easily.

logger = logging.getLogger(__name__)

mcp = FastMCP("AgenticDriving")

# Global references to simulation components
carla_client_instance = None
world_state_extractor_instance = None
action_executor_instance = None

def init_mcp_server(client, world_state, action_executor):
    global carla_client_instance, world_state_extractor_instance, action_executor_instance
    carla_client_instance = client
    world_state_extractor_instance = world_state
    action_executor_instance = action_executor

@mcp.tool()
def get_world_state() -> str:
    """
    Retrieves the current world state from the CARLA simulation,
    including ego vehicle speed/location and nearby actors.
    """
    if not world_state_extractor_instance:
        return json.dumps({"error": "Simulation not initialized"})
    
    state = world_state_extractor_instance.get_state()
    return json.dumps(state)

@mcp.tool()
def execute_action(action: str) -> str:
    """
    Executes a discrete driving action in the CARLA simulation.
    Valid actions: overtake, follow_lane, stop, yield, change_lane_left, change_lane_right.
    """
    if not action_executor_instance:
        return json.dumps({"error": "Simulation not initialized"})
        
    success = action_executor_instance.execute_action(action)
    if success:
        return json.dumps({"status": "success", "action": action})
    else:
        return json.dumps({"status": "error", "message": f"Failed to execute action: {action}"})

def run_mcp_server():
    """Run the MCP server."""
    mcp.run()
