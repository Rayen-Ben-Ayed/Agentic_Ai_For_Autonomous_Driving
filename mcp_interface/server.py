from typing import Literal

from mcp.server.fastmcp import FastMCP
import json
import logging

from pipeline_log import log_stage, summarize_world_state
from simulation import step_context
from simulation.junction_planner import ACTION_TO_DIRECTION, JUNCTION_ACTIONS
from simulation.maneuver_policy import (
    LATERAL_ACTIONS,
    PROACTIVE_ACTIONS,
    compute_allowed_actions,
    is_action_allowed,
    is_stuck_mode,
)
from simulation.lane_change_controller import LANE_CHANGE_LEAD_MARGIN_M, merge_lead_ok
from simulation import lane_controller as lc

DrivingAction = Literal[
    "overtake",
    "follow_lane",
    "stop",
    "yield",
    "change_lane_left",
    "change_lane_right",
    "go_straight",
    "turn_right",
    "turn_left",
]

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
    state["allowed_actions"] = compute_allowed_actions(state, stuck=stuck)
    return state


def _lane_preference_allows(action: str, state: dict) -> bool:
    return bool(state.get("lane_preference_allowed")) and action == state.get(
        "preferred_action"
    )


def _validate_action(action: str, state: dict) -> str | None:
    if state.get("error"):
        return json.dumps({"error": state["error"]})

    ego_speed = _live_ego_speed()
    stuck = is_stuck_mode(ego_speed, step_context.collisions_this_step())
    closest = state.get("effective_closest_distance") or state.get("closest_ahead_distance")

    if is_action_allowed(action, state, stuck=stuck):
        return None

    if stuck:
        return _reject_action(
            action,
            "Vehicle stuck after contact. Use stop or yield only.",
            state,
        )

    if action == "follow_lane" and state.get("too_close_for_follow_lane"):
        return _reject_action(
            action,
            f"Too close ({closest}m) to continue with throttle. Use yield or stop.",
            state,
        )

    if action in JUNCTION_ACTIONS:
        if not state.get("junction_ahead"):
            return _reject_action(
                action, "No junction ahead — junction actions unavailable.", state
            )
        if state.get("junction_committed"):
            committed_dir = state.get("junction_committed_direction")
            return _reject_action(
                action,
                (
                    f"Direction already committed ({committed_dir}) — no need to "
                    "re-issue it. Continue with follow_lane/yield/stop (it steers "
                    "the committed path automatically), or change_lane_* to bail out."
                ),
                state,
            )
        preferred = state.get("junction_preferred_action")
        options = state.get("junction_options") or {}
        if preferred is None:
            return _reject_action(
                action,
                "The junction ahead has no usable exit. Stop before it.",
                state,
            )
        if action != preferred:
            direction = ACTION_TO_DIRECTION.get(action)
            lane_note = ""
            if state.get("on_leftmost_lane") and direction == "right":
                lane_note = " Right turn is forbidden from the leftmost lane."
            elif state.get("on_rightmost_lane") and direction == "left":
                lane_note = " Left turn is forbidden from the rightmost lane."
            return _reject_action(
                action,
                (
                    f"No usable {direction} exit or a higher-priority lane-legal exit exists "
                    f"(order: forward > right > left; options={options}; "
                    f"on_leftmost_lane={state.get('on_leftmost_lane')}, "
                    f"on_rightmost_lane={state.get('on_rightmost_lane')}).{lane_note} "
                    f"Use {preferred}."
                ),
                state,
            )
        if state.get("too_close_for_follow_lane"):
            return _reject_action(
                action,
                f"Hazard {closest}m ahead — yield or stop before turning.",
                state,
            )

    # No exit exists at all (dead end / a junction with no usable branch):
    # only yield/stop can resolve it. An ordinary passable junction does NOT
    # restrict follow_lane or lane changes — those are handled normally,
    # tracking whatever direction was (or wasn't yet) committed.
    no_exit_ahead = (
        state.get("junction_ahead") and state.get("junction_preferred_action") is None
    ) or state.get("road_end_ahead")
    if state.get("junction_imminent") and no_exit_ahead:
        if action == "follow_lane" or action in LATERAL_ACTIONS:
            return _reject_action(
                action,
                (
                    "No exit at the upcoming junction/road end — "
                    f"{action} cannot resolve it. Use yield, then stop."
                ),
                state,
            )

    path_blocked = state.get("path_blocked", False)
    if action in PROACTIVE_ACTIONS:
        if not path_blocked and not _lane_preference_allows(action, state):
            return _reject_action(
                action,
                "Path is clear: use follow_lane or change_lane_right when "
                "lane_preference_allowed (keep-right discipline).",
                state,
            )
        if path_blocked and not state.get("maneuver_allowed"):
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
        if not _lane_preference_allows(action, state):
            if state.get("maneuver_too_close_for_lane_change"):
                return _reject_action(
                    action,
                    f"Too close for lateral move ({closest}m). Use yield or stop.",
                    state,
                )
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

    return _reject_action(action, "Action not allowed in current state.", state)


def init_mcp_server(client, world_state, action_executor):
    global carla_client_instance, world_state_extractor_instance, action_executor_instance
    carla_client_instance = client
    world_state_extractor_instance = world_state
    action_executor_instance = action_executor


def _action_feasibility(action: str, geometry: dict, state: dict) -> dict:
    """Cross-check a resolved action's geometry against the world state."""
    stuck = bool(state.get("stuck"))
    allowed = is_action_allowed(action, state, stuck=stuck)
    reasons: list[str] = []
    if not allowed:
        reasons.append("not in allowed_actions for the current state")

    effective_closest = state.get("effective_closest_distance")

    if action in JUNCTION_ACTIONS:
        if not geometry.get("junction_ahead", False):
            reasons.append("no junction ahead within detection range")
        elif not geometry.get("option_available", False):
            direction = geometry.get("direction")
            reasons.append(f"the junction ahead has no {direction} exit")
        if state.get("junction_committed") and not geometry.get("already_committed"):
            reasons.append(
                f"direction already committed ({state.get('junction_committed_direction')}); "
                "use follow_lane/yield/stop instead of re-deciding"
            )
        preferred = state.get("junction_preferred_action")
        if preferred and action != preferred:
            reasons.append(
                f"exit priority is forward > right > left among lane-legal exits: use {preferred}"
            )
        if (
            state.get("path_blocked")
            and effective_closest is not None
            and geometry.get("junction_distance_m") is not None
            and effective_closest < geometry["junction_distance_m"]
        ):
            reasons.append(
                f"an obstacle {effective_closest}m ahead blocks the approach "
                f"to the junction ({geometry['junction_distance_m']}m)"
            )

    if action in LATERAL_ACTIONS:
        if not geometry.get("target_lane_available", False):
            reasons.append("no adjacent driving lane on the target side")
        side = geometry.get("target_side")
        clear_flag = {
            "left": state.get("left_lane_clear"),
            "right": state.get("right_lane_clear"),
        }.get(side)
        if side is not None and clear_flag is False:
            reasons.append(f"{side} lane is not clear")
        merge_distance = geometry.get("merge_distance_m")
        if (
            state.get("path_blocked")
            and effective_closest is not None
            and merge_distance is not None
        ):
            fits = merge_lead_ok(merge_distance, effective_closest)
            geometry["merge_fits_before_hazard"] = fits
            if not fits:
                margin = merge_distance + LANE_CHANGE_LEAD_MARGIN_M
                reasons.append(
                    f"merge needs {merge_distance}m + lead margin but hazard is "
                    f"{effective_closest}m away (need ~{margin:.1f}m)"
                )
        if action_executor_instance and action_executor_instance.is_lane_centering_active():
            if not _lane_preference_allows(action, state):
                ego = action_executor_instance.carla_client.get_ego_vehicle()
                if ego is not None:
                    lat = action_executor_instance.frozen_centering_lat_err_m(ego)
                    if lat is not None and abs(lat) > lc.CENTER_TOLERANCE_M:
                        reasons.append(
                            f"previous lane change not centered (lat_err={lat:.2f}m)"
                        )

    feasible = allowed and not reasons
    return {"feasible": feasible, "reasons": reasons}


@mcp.tool()
def get_world_state() -> str:
    """Retrieves the current world state from the CARLA simulation."""
    state = _annotate_runtime_status(dict(_state_for_agent()))
    log_stage(logger, "MCP-server", "get_world_state -> %s", summarize_world_state(state))
    return json.dumps(state)


@mcp.tool()
def preview_action(action: DrivingAction) -> str:
    """Resolve a driving action into concrete waypoints and merge geometry.

    Returns what the action means as a trajectory (target lane, lateral offset,
    merge distance/time, target speed) together with a feasibility cross-check
    against the current world state. Call this before execute_action: it is
    mandatory for every action except `stop`, and it does NOT move the vehicle.
    """
    if not action_executor_instance:
        return json.dumps({"error": "Simulation not initialized"})

    state = _annotate_runtime_status(dict(_state_for_agent()))
    geometry = action_executor_instance.describe_action(action)
    if geometry.get("error"):
        return json.dumps({"action": action, **geometry})

    feasibility = _action_feasibility(action, geometry, state)
    step_context.record_preview(action)

    payload = {
        "status": "preview",
        **geometry,
        **feasibility,
        "world": {
            "path_blocked": state.get("path_blocked"),
            "effective_closest_distance": state.get("effective_closest_distance"),
            "maneuver_allowed": state.get("maneuver_allowed"),
            "lane_change_allowed": state.get("lane_change_allowed"),
            "left_lane_clear": state.get("left_lane_clear"),
            "right_lane_clear": state.get("right_lane_clear"),
            "junction_ahead": state.get("junction_ahead"),
            "junction_imminent": state.get("junction_imminent"),
            "junction_options": state.get("junction_options"),
            "junction_preferred_action": state.get("junction_preferred_action"),
            "junction_committed": state.get("junction_committed"),
            "junction_committed_direction": state.get("junction_committed_direction"),
            "allowed_actions": state.get("allowed_actions"),
        },
    }
    log_stage(
        logger,
        "MCP-server",
        "preview_action %s -> feasible=%s merge_d=%s merge_t=%s tgt_lane=%s reasons=%s",
        action,
        feasibility["feasible"],
        geometry.get("merge_distance_m"),
        geometry.get("merge_duration_s"),
        geometry.get("target_lane_id"),
        feasibility["reasons"],
    )
    return json.dumps(payload)


@mcp.tool()
def execute_action(action: DrivingAction) -> str:
    """Executes a discrete driving action in the CARLA simulation."""
    if not action_executor_instance:
        return json.dumps({"error": "Simulation not initialized"})

    state = _annotate_runtime_status(dict(_state_for_agent()))
    rejection = _validate_action(action, state)
    if rejection:
        return rejection

    # Mandatory preview gating: the agent must cross-check an action with
    # preview_action before executing it. `stop` is exempt so the deterministic
    # safety net can always apply a defensive command.
    if action != "stop" and not step_context.was_previewed(action):
        return _reject_action(
            action,
            "Call preview_action(action) to cross-check this action against the "
            "world state before executing it.",
            state,
        )

    success = action_executor_instance.execute_action(action)
    if success:
        log_stage(logger, "MCP-server", "execute_action OK -> %s", action)
        return json.dumps({"status": "success", "action": action})
    log_stage(logger, "MCP-server", "execute_action FAILED -> %s", action)
    return json.dumps({"status": "error", "message": f"Failed to execute action: {action}"})


def run_mcp_server():
    mcp.run()
