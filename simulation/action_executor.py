"""Translates discrete agent actions into Phabmacs bridge requests."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ActionExecutor:
    VALID_ACTIONS = [
        "overtake",
        "follow_lane",
        "stop",
        "yield",
        "change_lane_left",
        "change_lane_right",
    ]

    def __init__(self, bridge):
        self.bridge = bridge

    def execute_action(self, action: str) -> bool:
        if action not in self.VALID_ACTIONS:
            logger.error("Invalid action %r. Valid: %s", action, self.VALID_ACTIONS)
            return False
        ok = self.bridge.send_action(action)
        if ok:
            logger.info("Action %s queued in Phabmacs", action)
        return ok
