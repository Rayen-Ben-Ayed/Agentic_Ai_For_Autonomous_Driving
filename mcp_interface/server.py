"""MCP server exposing the Phabmacs simulation (optional; not used by main.py)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from simulation.agent_tools import execute_action as _execute_action
from simulation.agent_tools import get_world_state as _get_world_state
from simulation.agent_tools import init_agent_tools

mcp = FastMCP("AgenticDriving")


def init_mcp_server(bridge, world_state, action_executor):
    init_agent_tools(bridge, world_state, action_executor)


@mcp.tool()
def get_world_state() -> str:
    """Returns the current ego/world state from Phabmacs as a JSON string."""
    return _get_world_state()


@mcp.tool()
def execute_action(action: str) -> str:
    """Execute a discrete driving action in Phabmacs."""
    return _execute_action(action)


def run_mcp_server():
    mcp.run()
