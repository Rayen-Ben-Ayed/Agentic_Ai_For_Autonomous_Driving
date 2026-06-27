import time
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self):
        self.collisions = 0
        self.decision_latencies = []
        self.rule_violations = 0
        self.start_time = None
        self.first_collision_actor = None

    def start_decision_timer(self):
        self.start_time = time.time()

    def end_decision_timer(self):
        if self.start_time:
            latency = (time.time() - self.start_time) * 1000
            self.decision_latencies.append(latency)
            self.start_time = None
            return latency
        return 0

    def record_collision(self, other_actor_type: str | None = None):
        """Record a single raw contact substep.

        CARLA fires the collision sensor on every physics substep of contact, so
        this is a contact-substep counter (used by stuck detection), NOT the
        number of distinct crash events. Discrete events are tracked separately
        via the collision burst gate in the Evaluator.
        """
        if self.collisions == 0 and other_actor_type:
            self.first_collision_actor = other_actor_type
        self.collisions += 1

    def record_rule_violation(self):
        self.rule_violations += 1
        logger.warning("Rule violation recorded!")

    def get_summary(self):
        avg_latency = (
            sum(self.decision_latencies) / len(self.decision_latencies)
            if self.decision_latencies
            else 0
        )
        return {
            "total_contact_substeps": self.collisions,
            "first_collision_with": self.first_collision_actor,
            "average_latency_ms": round(avg_latency, 2),
            "max_latency_ms": (
                round(max(self.decision_latencies), 2) if self.decision_latencies else 0
            ),
            "rule_violations": self.rule_violations,
            "success": self.collisions == 0,
        }
