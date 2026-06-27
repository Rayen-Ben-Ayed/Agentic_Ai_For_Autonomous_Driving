"""
Standalone MCP server entry point (stdio) for external MCP clients.

CARLA must be initialized in the same process before tools can return real data.
For the built-in driving loop, use main.py instead — it wires CARLA and runs
the in-process MCPDrivingClient.
"""
from mcp_interface.server import mcp, run_mcp_server

if __name__ == "__main__":
    run_mcp_server()
