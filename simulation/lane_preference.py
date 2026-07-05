"""Keep-right lane discipline: prefer the rightmost travel lane when path is clear."""
from __future__ import annotations

import os

from simulation.maneuver_policy import compute_allowed_actions

KEEP_RIGHT_ENABLED = os.getenv("KEEP_RIGHT_ENABLED", "1").lower() in ("1", "true", "yes")


def enrich_keep_right_preference(state: dict) -> None:
    """Set lane-preference fields and refresh ``allowed_actions`` on ``state``."""
    ego = state.get("ego_vehicle") or {}
    on_rightmost = state.get("on_rightmost_lane")
    if on_rightmost is None:
        on_rightmost = ego.get("on_rightmost_lane", True)

    path_blocked = bool(state.get("path_blocked", False))
    right_clear = bool(state.get("right_lane_clear", False))
    right_avail = bool(
        state.get("right_lane_available", ego.get("right_lane_available", False))
    )

    # Round 1 of the junction decision (direction not yet committed) outranks
    # keep-right lane discipline; once committed, follow_lane/yield/stop drive
    # it (round 2) and normal lane preference resumes.
    junction_ahead = bool(state.get("junction_ahead"))
    junction_committed = bool(state.get("junction_committed"))
    junction_pending = junction_ahead and not junction_committed
    preference_allowed = (
        KEEP_RIGHT_ENABLED
        and not path_blocked
        and not on_rightmost
        and right_clear
        and right_avail
        and not junction_pending
    )

    state["lane_discipline"] = "keep_right"
    state["on_rightmost_lane"] = on_rightmost
    state["lane_preference_allowed"] = preference_allowed
    junction_action = state.get("junction_preferred_action")
    caution = state.get("preferred_caution_action")
    if caution and state.get("too_close_for_follow_lane"):
        state["preferred_action"] = caution
    elif junction_pending and junction_action:
        state["preferred_action"] = junction_action
    elif preference_allowed:
        state["preferred_action"] = "change_lane_right"
    else:
        state["preferred_action"] = "follow_lane"
    state["allowed_actions"] = compute_allowed_actions(
        state, stuck=bool(state.get("stuck", False))
    )
