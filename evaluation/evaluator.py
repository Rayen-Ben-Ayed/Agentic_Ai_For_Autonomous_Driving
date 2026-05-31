"""Evaluation pipeline for the Phabmacs-based agent.

Collisions are observed via the bridge (which mirrors VehicleCollisionEvent from
the simulator). Decision latency and rule violations are tracked locally.
"""

from __future__ import annotations

import json
import logging

from evaluation.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, bridge):
        self.bridge = bridge
        self.metrics = MetricsTracker()
        self._last_collision_count = 0

    def setup_sensors(self) -> None:
        """No-op for Phabmacs (collisions are surfaced through the bridge)."""
        self._last_collision_count = self.bridge.get_metrics().get("collisions", 0)
        logger.info("Evaluator armed; initial collision count = %d", self._last_collision_count)

    def poll_collisions(self) -> None:
        """Should be called periodically by the main loop."""
        m = self.bridge.get_metrics()
        c = int(m.get("collisions", 0))
        delta = c - self._last_collision_count
        for _ in range(max(delta, 0)):
            self.metrics.record_collision()
        self._last_collision_count = c

    def cleanup(self) -> None:
        pass

    def log_results(self, filepath: str = "evaluation_results.json") -> None:
        summary = self.metrics.get_summary()
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info("Evaluation results saved to %s", filepath)
        logger.info("Summary: %s", summary)
