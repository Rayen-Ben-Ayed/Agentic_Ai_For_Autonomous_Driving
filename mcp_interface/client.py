import asyncio
import json
import logging
from typing import Any, Optional

from mcp_interface.server import mcp
from pipeline_log import log_stage, log_tool_result

logger = logging.getLogger(__name__)


def _tool_result_to_text(result: Any) -> str:
    """Normalize FastMCP call_tool return value to a string for the LLM."""
    if isinstance(result, tuple):
        blocks, meta = result[0], result[1] if len(result) > 1 else None
        if meta and isinstance(meta, dict) and "result" in meta:
            return str(meta["result"])
        if blocks:
            parts = []
            for block in blocks:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
    if isinstance(result, dict):
        return json.dumps(result)
    return str(result)


class MCPDrivingClient:
    """
    In-process MCP client for the driving agent.
    Routes LLM tool calls through the FastMCP server (get_world_state, execute_action).
    """

    def __init__(self, mcp_server=mcp, verbose_state: bool = False, benchmark_collector=None):
        self._mcp = mcp_server
        self._openai_tools: Optional[list[dict]] = None
        self._verbose_state = verbose_state
        self._benchmark_collector = benchmark_collector

    async def list_tools(self):
        return await self._mcp.list_tools()

    async def call_tool(self, name: str, arguments: Optional[dict] = None) -> str:
        arguments = arguments or {}
        if self._benchmark_collector is not None:
            self._benchmark_collector.record_mcp_tool_call(name, arguments)
        log_stage(logger, "MCP-client", "call_tool %s(%s)", name, arguments)
        result = await self._mcp.call_tool(name, arguments)
        text = _tool_result_to_text(result)
        log_tool_result(logger, name, text, verbose=self._verbose_state)
        return text

    async def get_world_state(self) -> dict:
        text = await self.call_tool("get_world_state", {})
        return json.loads(text)

    async def execute_action(self, action: str) -> dict:
        text = await self.call_tool("execute_action", {"action": action})
        return json.loads(text)

    async def get_openai_tools(self) -> list[dict]:
        if self._openai_tools is not None:
            return self._openai_tools

        mcp_tools = await self.list_tools()
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": (tool.description or "").strip(),
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )
        self._openai_tools = openai_tools
        return openai_tools

    def get_openai_tools_sync(self) -> list[dict]:
        return asyncio.run(self.get_openai_tools())

    def call_tool_sync(self, name: str, arguments: Optional[dict] = None) -> str:
        return asyncio.run(self.call_tool(name, arguments))

    def get_world_state_sync(self) -> dict:
        return asyncio.run(self.get_world_state())
