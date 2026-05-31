"""Deterministic fallback when no LLM API key is available (--mock)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def decide_overtake(state: Dict[str, Any]) -> str:
    """Simple overtaking policy using bridge traffic fields."""
    if state.get("error"):
        logger.warning("State error: %s", state["error"])
        return "follow_lane"

    traffic = state.get("traffic") or {}
    left_clear = traffic.get("left_lane_clear", True)
    right_busy = traffic.get("right_lane_occupied", False)
    slow_ahead = traffic.get("slow_vehicle_ahead", False)
    dist_front = traffic.get("distance_to_front")

    front = state.get("front_vehicle")
    if front and isinstance(front, dict):
        d = front.get("distance")
        if d is not None and d < 75.0:
            slow_ahead = True
            dist_front = d

    if slow_ahead and left_clear and dist_front is not None and dist_front < 80.0:
        logger.info("Rule agent: overtake (dist=%.1f)", dist_front)
        return "overtake"

    if slow_ahead and left_clear:
        return "change_lane_left"

    if right_busy:
        return "follow_lane"

    return "follow_lane"
