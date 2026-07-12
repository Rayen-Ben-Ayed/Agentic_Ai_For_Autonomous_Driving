"""
Speed- and latency-aware rules for when proactive maneuvers (lane change) make sense.
"""
import os
from typing import Optional

from simulation.junction_planner import JUNCTION_ACTIONS
from simulation.timing_config import AGENT_LATENCY_S, STEP_INTERVAL_S
SAFETY_MARGIN_M = float(os.getenv("MANEUVER_SAFETY_MARGIN_M", "5.0"))
MAX_MANEUVER_TRIGGER_M = float(os.getenv("MAX_MANEUVER_TRIGGER_M", "40.0"))
MIN_MANEUVER_HORIZON_M = float(os.getenv("MIN_MANEUVER_HORIZON_M", "12.0"))
LANE_CHANGE_MIN_DIST_M = float(os.getenv("LANE_CHANGE_MIN_DIST_M", "10.0"))

# Minimum speed used only for horizon math while ego is accelerating from rest
MIN_PLANNING_SPEED_MPS = float(os.getenv("MIN_PLANNING_SPEED_MPS", "3.5"))

# Below this gap, follow_lane with throttle is rejected (use yield/stop)
CLOSE_FOLLOW_MIN_DIST_M = float(os.getenv("CLOSE_FOLLOW_MIN_DIST_M", "12.0"))

STUCK_SPEED_MPS = float(os.getenv("STUCK_SPEED_MPS", "0.5"))

# Contacts in one step before treating ego as stuck (avoids single sensor spikes)
STUCK_COLLISION_DELTA = int(os.getenv("STUCK_COLLISION_DELTA", "5"))

ALL_DRIVING_ACTIONS = (
    "follow_lane",
    "stop",
    "yield",
    "change_lane_left",
    "change_lane_right",
    *JUNCTION_ACTIONS,
)

LATERAL_ACTIONS = frozenset({
    "change_lane_left",
    "change_lane_right",
})

PROACTIVE_ACTIONS = LATERAL_ACTIONS


def compute_maneuver_horizon_m(ego_speed_m_s: float) -> float:
    planning_speed = max(ego_speed_m_s, MIN_PLANNING_SPEED_MPS)
    reaction_time_s = AGENT_LATENCY_S + STEP_INTERVAL_S
    dynamic_horizon = planning_speed * reaction_time_s + SAFETY_MARGIN_M
    return min(max(dynamic_horizon, MIN_MANEUVER_HORIZON_M), MAX_MANEUVER_TRIGGER_M)


def _effective_closest(
    closest_ahead_m: Optional[float],
    closest_blocking_m: Optional[float],
) -> Optional[float]:
    values = [d for d in (closest_ahead_m, closest_blocking_m) if d is not None]
    return min(values) if values else None


def evaluate_maneuver_policy(
    obstacle_ahead: bool,
    closest_ahead_m: Optional[float],
    ego_speed_m_s: float,
    blocking_vehicle_ahead: bool = False,
    closest_blocking_m: Optional[float] = None,
) -> dict:
    effective_closest = _effective_closest(closest_ahead_m, closest_blocking_m)
    path_blocked = obstacle_ahead or blocking_vehicle_ahead
    horizon_m = compute_maneuver_horizon_m(ego_speed_m_s)
    planning_speed = max(ego_speed_m_s, MIN_PLANNING_SPEED_MPS)

    # An action is committed for one decision window (STEP_INTERVAL_S). To be
    # "futureproof" over that window, follow_lane must keep enough gap that the
    # ego cannot reach the obstacle before the next decision: the distance it
    # travels in the window plus a safety margin (never below the static floor).
    commit_travel_m = planning_speed * STEP_INTERVAL_S
    safe_follow_dist_m = max(CLOSE_FOLLOW_MIN_DIST_M, commit_travel_m + SAFETY_MARGIN_M)

    too_far = not path_blocked or effective_closest is None or effective_closest > horizon_m
    too_close_for_lateral = (
        effective_closest is not None and effective_closest < LANE_CHANGE_MIN_DIST_M
    )
    too_close_for_follow = (
        effective_closest is not None and effective_closest < safe_follow_dist_m
    )

    base_allowed = path_blocked and effective_closest is not None and not too_far

    return {
        "maneuver_horizon_m": round(horizon_m, 2),
        "planning_speed_mps": round(planning_speed, 2),
        "agent_reaction_time_s": round(AGENT_LATENCY_S + STEP_INTERVAL_S, 2),
        "decision_window_s": round(STEP_INTERVAL_S, 2),
        "follow_safe_distance_m": round(safe_follow_dist_m, 2),
        "effective_closest_distance": (
            round(effective_closest, 2) if effective_closest is not None else None
        ),
        "path_blocked": path_blocked,
        "maneuver_allowed": base_allowed,
        "lane_change_allowed": base_allowed and not too_close_for_lateral,
        "maneuver_too_far": path_blocked and too_far,
        "maneuver_too_close_for_lane_change": base_allowed and too_close_for_lateral,
        "too_close_for_follow_lane": path_blocked and too_close_for_follow,
        "prefer_yield_or_stop": path_blocked and too_close_for_follow,
    }


def _adjacent_lane_action_clear(action: str, state: dict) -> bool:
    """Whether the target side is clear enough for this lateral action."""
    if action == "change_lane_left":
        if not state.get("left_lane_available"):
            return False
        if state.get("left_lane_clear"):
            return True
        if (
            state.get("lane_centering_incomplete")
            and state.get("lane_centering_side") == "left"
        ):
            return True
        return False
    if action == "change_lane_right":
        if not state.get("right_lane_available"):
            return False
        if state.get("right_lane_clear"):
            return True
        if (
            state.get("lane_centering_incomplete")
            and state.get("lane_centering_side") == "right"
        ):
            return True
        return False
    return True


def is_stuck_mode(ego_speed_m_s: float, collisions_this_step: int) -> bool:
    return (
        ego_speed_m_s < STUCK_SPEED_MPS
        and collisions_this_step >= STUCK_COLLISION_DELTA
    )


def allowed_actions_when_stuck() -> frozenset[str]:
    return frozenset({"stop", "yield"})


def junction_round1_commit_pending(state: dict) -> bool:
    """Round 1 at a passable junction: commit a direction, don't park on a clear path."""
    if not state.get("junction_ahead") or state.get("junction_committed"):
        return False
    if not state.get("junction_preferred_action"):
        return False
    if state.get("path_blocked") or state.get("too_close_for_follow_lane"):
        return False
    return True


def is_action_allowed(action: str, state: dict, *, stuck: bool = False) -> bool:
    """Mirror MCP execute_action validation (positive form)."""
    if state.get("error"):
        return False

    if stuck:
        return action in allowed_actions_when_stuck()

    # A pedestrian is crossing the ego path. A lane change cannot escape
    # a walker crossing the whole carriageway — it only leaves the ego misaligned
    # (which then whips the junction tracker off-road). Force the braking
    # response: yield/stop only, until the crossing clears.
    if state.get("preferred_caution_action") in ("yield", "stop"):
        return action in ("yield", "stop")

    if junction_round1_commit_pending(state):
        return action == state.get("junction_preferred_action")

    path_blocked = state.get("path_blocked", False)
    preferred = state.get("preferred_action")
    junction_imminent = bool(state.get("junction_imminent"))

    # Two-round junction handling: round 1 is a one-time direction commit
    # (go_straight/turn_right/turn_left); round 2 is the ordinary action menu
    # below, which steers along whatever was committed. Once committed, the
    # junction actions themselves are no longer offered — re-deciding is not
    # needed, and it previously combined with a blanket follow_lane/lateral
    # veto to produce a dead end (no action left once an obstacle was merely
    # "not immediately close" near a junction).
    if action in JUNCTION_ACTIONS:
        if not state.get("junction_ahead"):
            return False
        if state.get("junction_committed"):
            return False  # already decided; drive it with follow_lane/yield/stop
        # Fixed priority (forward, then right, then left) among lane-legal exits:
        # only the single best available exit is offered as a junction action.
        if action != state.get("junction_preferred_action"):
            return False
        if state.get("too_close_for_follow_lane"):
            return False  # hazard immediately ahead: yield/stop first
        return True

    if action == "follow_lane" and state.get("too_close_for_follow_lane"):
        return False

    if action == "follow_lane" and state.get("lane_centering_incomplete"):
        return False

    # No exit exists at all (dead end, or a junction with no usable branch):
    # in-lane cruising and lateral merges cannot resolve it, only yield/stop
    # can. This is narrower than blocking on every junction — an ordinary,
    # passable junction never restricts follow_lane/lane-change this way.
    no_exit_ahead = (
        state.get("junction_ahead") and state.get("junction_preferred_action") is None
    ) or state.get("road_end_ahead")
    if junction_imminent and no_exit_ahead:
        if action == "follow_lane":
            return False
        if action in LATERAL_ACTIONS:
            return False

    # Lane preference when the path is clear (keep-right discipline).
    if (
        not path_blocked
        and state.get("lane_preference_allowed")
        and action == preferred
    ):
        if action == "follow_lane":
            return True
        if action in ("change_lane_left", "change_lane_right"):
            return _adjacent_lane_action_clear(action, state)

    if action in PROACTIVE_ACTIONS:
        if not path_blocked or not state.get("maneuver_allowed"):
            return False

    if action in LATERAL_ACTIONS and not state.get("lane_change_allowed"):
        preference_ok = (
            not path_blocked
            and state.get("lane_preference_allowed")
            and action == preferred
        )
        if not preference_ok:
            return False

    if action == "change_lane_left" and not _adjacent_lane_action_clear("change_lane_left", state):
        return False
    if action == "change_lane_right" and not _adjacent_lane_action_clear("change_lane_right", state):
        return False

    return True


def compute_allowed_actions(state: dict, *, stuck: bool = False) -> list[str]:
    if stuck:
        return sorted(allowed_actions_when_stuck())
    return sorted(
        action
        for action in ALL_DRIVING_ACTIONS
        if is_action_allowed(action, state, stuck=False)
    )
