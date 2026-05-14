import time
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self):
        self.collisions = 0
        self.decision_latencies = []
        self.rule_violations = 0
        self.start_time = None

    def start_decision_timer(self):
        self.start_time = time.time()

    def end_decision_timer(self):
        if self.start_time:
            latency = (time.time() - self.start_time) * 1000 # in ms
            self.decision_latencies.append(latency)
            self.start_time = None
            return latency
        return 0

    def record_collision(self):
        self.collisions += 1
        logger.warning("Collision recorded!")

    def record_rule_violation(self):
        self.rule_violations += 1
        logger.warning("Rule violation recorded!")

    def get_summary(self):
        avg_latency = sum(self.decision_latencies) / len(self.decision_latencies) if self.decision_latencies else 0
        return {
            "total_collisions": self.collisions,
            "average_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(max(self.decision_latencies), 2) if self.decision_latencies else 0,
            "rule_violations": self.rule_violations,
            "success": self.collisions == 0
        }
