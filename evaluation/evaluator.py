import json
import logging
import carla
from evaluation.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, carla_client):
        self.carla_client = carla_client
        self.metrics = MetricsTracker()
        self.collision_sensor = None

    def setup_sensors(self):
        """
        Sets up the collision sensor on the ego vehicle to track collisions.
        """
        world = self.carla_client.get_world()
        ego_vehicle = self.carla_client.get_ego_vehicle()
        
        if not world or not ego_vehicle:
            logger.error("Cannot setup sensors: world or ego vehicle not initialized.")
            return

        blueprint_library = world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        
        self.collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        logger.info("Collision sensor attached.")

    def _on_collision(self, event):
        self.metrics.record_collision()
        logger.error(f"Collision with {event.other_actor.type_id}")

    def cleanup(self):
        if self.collision_sensor:
            self.collision_sensor.destroy()
            logger.info("Collision sensor destroyed.")

    def log_results(self, filepath="evaluation_results.json"):
        summary = self.metrics.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Evaluation results saved to {filepath}")
        logger.info(f"Summary: {summary}")
