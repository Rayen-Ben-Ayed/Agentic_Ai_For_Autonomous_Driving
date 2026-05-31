"""Direct simulation tools for the decision loop (no FastMCP import).

decision_maker imports from here so `python main.py` starts quickly and does not
depend on the MCP server package at runtime.
"""

from __future__ import annotations

import json
from typing import Optional

_bridge = None
_world_state_extractor = None
_action_executor = None


def init_agent_tools(bridge, world_state, action_executor) -> None:
    global _bridge, _world_state_extractor, _action_executor
    _bridge = bridge
    _world_state_extractor = world_state
    _action_executor = action_executor


def get_world_state() -> str:
    if _world_state_extractor is None:
        return json.dumps({"error": "Simulation not initialized"})
    return json.dumps(_world_state_extractor.get_state())


def execute_action(action: str) -> str:
    if _action_executor is None:
        return json.dumps({"error": "Simulation not initialized"})
    ok = _action_executor.execute_action(action)
    if ok:
        return json.dumps({"status": "success", "action": action})
    return json.dumps({"status": "error", "message": f"Failed to execute action: {action}"})


def get_world_state_dict() -> dict:
    raw = get_world_state()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid state json"}
