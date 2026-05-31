"""World state extractor for the Phabmacs bridge.

The bridge already returns JSON shaped almost like the structure expected by the
LLM prompt, so this class is a thin pass-through with a small reshaping for
backwards compatibility with the CARLA-era schema.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class WorldStateExtractor:
    def __init__(self, bridge):
        self.bridge = bridge

    def get_state(self) -> Dict[str, Any]:
        raw = self.bridge.get_state()
        if "error" in raw:
            return raw

        ego = raw.get("ego_vehicle", {})
        loc = ego.get("location", {})
        head = ego.get("heading", {})
        front = raw.get("front_vehicle")
        surr = raw.get("surroundings", {})

        nearby = []
        if isinstance(front, dict):
            nearby.append({
                "id": "front",
                "type": "vehicle",
                "distance": front.get("distance"),
                "speed": front.get("speed"),
                "position": "FRONT",
            })
        for pos_name, present in surr.items():
            if present and pos_name != "FRONT":
                nearby.append({
                    "id": pos_name.lower(),
                    "type": "vehicle",
                    "position": pos_name,
                })

        return {
            "scenario": raw.get("scenario", "default"),
            "scenario_hint": raw.get("scenario_hint"),
            "traffic": raw.get("traffic", {}),
            "ego_vehicle": {
                "speed": ego.get("speed"),
                "location": loc,
                "heading": head,
                "lane_change_in_progress": ego.get("lane_change_in_progress", False),
            },
            "front_vehicle": raw.get("front_vehicle"),
            "surroundings": surr,
            "nearby_actors": nearby,
            "collisions": raw.get("collisions", 0),
            "current_action": raw.get("current_action"),
        }
