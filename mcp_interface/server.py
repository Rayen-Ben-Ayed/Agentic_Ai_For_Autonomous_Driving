from typing import Literal

from mcp.server.fastmcp import FastMCP
import json
import logging

from pipeline_log import log_stage, summarize_world_state
from simulation import step_context
from simulation.maneuver_policy import (
    allowed_actions_when_stuck,
    is_stuck_mode,
)

DrivingAction = Literal[
    "overtake",
    "follow_lane",
    "stop",
    "yield",
    "change_lane_left",
    "change_lane_right",
]

LATERAL_ACTIONS = frozenset({
    "overtake",
    "change_lane_left",
    "change_lane_right",
})

PROACTIVE_ACTIONS = LATERAL_ACTIONS

logger = logging.getLogger(__name__)

mcp = FastMCP("AgenticDriving")

carla_client_instance = None
world_state_extractor_instance = None
action_executor_instance = None


def _reject_action(action: str, message: str, state: dict) -> str:
    logger.warning("Rejected %s: %s", action, message)
    return json.dumps({
        "status": "rejected",
        "message": message,
        "stuck": state.get("stuck"),
        "allowed_actions": state.get("allowed_actions"),
        "obstacle_ahead": state.get("obstacle_ahead"),
        "blocking_vehicle_ahead": state.get("blocking_vehicle_ahead"),
        "closest_ahead_distance": state.get("closest_ahead_distance"),
        "closest_blocking_distance": state.get("closest_blocking_distance"),
        "effective_closest_distance": state.get("effective_closest_distance"),
        "maneuver_horizon_m": state.get("maneuver_horizon_m"),
        "maneuver_allowed": state.get("maneuver_allowed"),
        "lane_change_allowed": state.get("lane_change_allowed"),
        "path_blocked": state.get("path_blocked"),
        "requested_action": action,
    })


def _state_for_agent() -> dict:
    frozen = step_context.get_frozen_state()
    if frozen is not None:
        return frozen
    if world_state_extractor_instance:
        return world_state_extractor_instance.get_state()
    return {"error": "Simulation not initialized"}


def _live_ego_speed() -> float:
    if not world_state_extractor_instance:
        return 0.0
    live = world_state_extractor_instance.get_state()
    return float((live.get("ego_vehicle") or {}).get("speed") or 0.0)


def _annotate_runtime_status(state: dict) -> dict:
    """Add live collision/stuck info so the agent can observe why an action was
    rejected. The geometry in `state` may be a frozen per-step snapshot, but
    these status flags always reflect the current simulation."""
    ego_speed = _live_ego_speed()
    collisions = step_context.collisions_this_step()
    stuck = is_stuck_mode(ego_speed, collisions)
    state["collisions_this_step"] = collisions
    state["stuck"] = stuck
    if stuck:
        state["allowed_actions"] = sorted(allowed_actions_when_stuck())
    return state


def _validate_action(action: str, state: dict) -> str | None:
    if state.get("error"):
        return json.dumps({"error": state["error"]})

    ego_speed = _live_ego_speed()
    stuck = is_stuck_mode(ego_speed, step_context.collisions_this_step())
    if stuck and action not in allowed_actions_when_stuck():
        return _reject_action(
            action,
            "Vehicle stuck after contact. Use stop or yield only.",
            state,
        )

    closest = state.get("effective_closest_distance") or state.get("closest_ahead_distance")
    path_blocked = state.get("path_blocked", False)

    if action == "follow_lane":
        if stuck:
            return _reject_action(
                action,
                "Vehicle stuck after contact. Use stop or yield.",
                state,
            )
        if state.get("too_close_for_follow_lane"):
            return _reject_action(
                action,
                f"Too close ({closest}m) to continue with throttle. Use yield or stop.",
                state,
            )

    if action in PROACTIVE_ACTIONS:
        if not path_blocked:
            return _reject_action(
                action,
                "Path is clear (no in-lane or blocking vehicle ahead). Use follow_lane.",
                state,
            )
        if not state.get("maneuver_allowed"):
            horizon = state.get("maneuver_horizon_m")
            return _reject_action(
                action,
                (
                    f"Too early for maneuver: effective distance {closest}m, "
                    f"horizon={horizon}m. Use follow_lane or yield."
                ),
                state,
            )

    if action in LATERAL_ACTIONS and not state.get("lane_change_allowed"):
        if state.get("maneuver_too_close_for_lane_change"):
            return _reject_action(
                action,
                f"Too close for lateral move ({closest}m). Use yield or stop.",
                state,
            )
        if not state.get("maneuver_allowed"):
            return _reject_action(
                action,
                "Lateral maneuver not allowed for current distance/speed.",
                state,
            )

    if action == "change_lane_left" and not state.get("left_lane_clear"):
        return _reject_action(action, "left_lane_clear is false.", state)
    if action == "change_lane_right" and not state.get("right_lane_clear"):
        return _reject_action(action, "right_lane_clear is false.", state)
    if action == "overtake" and not (
        state.get("left_lane_clear") or state.get("right_lane_clear")
    ):
        return _reject_action(action, "Both adjacent lanes blocked.", state)

    return None


def init_mcp_server(client, world_state, action_executor):
    global carla_client_instance, world_state_extractor_instance, action_executor_instance
    carla_client_instance = client
    world_state_extractor_instance = world_state
    action_executor_instance = action_executor


@mcp.tool()
def get_world_state() -> str:
    """Retrieves the current world state from the CARLA simulation."""
    state = _annotate_runtime_status(dict(_state_for_agent()))
    log_stage(logger, "MCP-server", "get_world_state -> %s", summarize_world_state(state))
    return json.dumps(state)


@mcp.tool()
def execute_action(action: DrivingAction) -> str:
    """Executes a discrete driving action in the CARLA simulation."""
    if not action_executor_instance:
        return json.dumps({"error": "Simulation not initialized"})

    state = _annotate_runtime_status(dict(_state_for_agent()))
    rejection = _validate_action(action, state)
    if rejection:
        return rejection

    success = action_executor_instance.execute_action(action)
    if success:
        log_stage(logger, "MCP-server", "execute_action OK -> %s", action)
        return json.dumps({"status": "success", "action": action})
    log_stage(logger, "MCP-server", "execute_action FAILED -> %s", action)
    return json.dumps({"status": "error", "message": f"Failed to execute action: {action}"})


def run_mcp_server():
    mcp.run()
