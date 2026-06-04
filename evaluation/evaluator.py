import json
import logging
import carla

from evaluation.metrics import MetricsTracker
from evaluation.collision_log import CollisionLogGate
from simulation import step_context

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, carla_client):
        self.carla_client = carla_client
        self.metrics = MetricsTracker()
        self.collision_sensor = None
        self.collision_log = CollisionLogGate()

    def setup_sensors(self):
        world = self.carla_client.get_world()
        ego_vehicle = self.carla_client.get_ego_vehicle()

        if not world or not ego_vehicle:
            logger.error("Cannot setup sensors: world or ego vehicle not initialized.")
            return

        blueprint_library = world.get_blueprint_library()
        collision_bp = blueprint_library.find("sensor.other.collision")

        self.collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego_vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        logger.info("Collision sensor attached.")

    def _on_collision(self, event):
        actor_type = event.other_actor.type_id
        self.metrics.record_collision(actor_type)
        step_context.update_live_collision_count(self.metrics.collisions)
        self.collision_log.record(actor_type)

    def cleanup(self):
        self.collision_log.finalize()
        if self.collision_sensor:
            self.collision_sensor.destroy()
            logger.info("Collision sensor destroyed.")

    def log_results(self, filepath="evaluation_results.json"):
        summary = self.metrics.get_summary()
        # Discrete crash events (deduplicated bursts) are the authoritative
        # safety metric; the raw substep count stays for diagnostics only.
        collision_events = self.collision_log.burst_count
        summary["collision_events"] = collision_events
        summary["success"] = collision_events == 0
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info("Evaluation results saved to %s", filepath)
        logger.info("Summary: %s", summary)
