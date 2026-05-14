import carla
import logging
from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)

class Scenario01Braking(BaseScenario):
    def setup(self):
        """
        Spawns a vehicle directly in front of the ego vehicle.
        """
        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not found. Cannot setup scenario.")
            return

        # Get ego vehicle's location and rotation
        ego_transform = ego_vehicle.get_transform()
        
        # Calculate a point 20 meters ahead of the ego vehicle
        forward_vector = ego_transform.get_forward_vector()
        spawn_location = ego_transform.location + (forward_vector * 20.0)
        spawn_transform = carla.Transform(spawn_location, ego_transform.rotation)

        # Spawn the NPC vehicle
        blueprint_library = self.world.get_blueprint_library()
        npc_bp = blueprint_library.filter('vehicle.audi.tt')[0]
        
        npc_vehicle = self.world.try_spawn_actor(npc_bp, spawn_transform)
        
        if npc_vehicle:
            self.npc_actors.append(npc_vehicle)
            logger.info("Scenario 01 Setup: Spawned NPC vehicle ahead.")
            
            # Make the NPC brake hard immediately
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            npc_vehicle.apply_control(control)
        else:
            logger.error("Scenario 01 Setup: Failed to spawn NPC vehicle.")

    def is_llm_needed(self, world_state):
        """
        Trigger the LLM if there is a vehicle within 25 meters ahead.
        We only want to query the LLM once for this specific event.
        """
        if self.llm_queried:
            return False # Already asked the LLM

        # Check nearby actors from the world state
        for actor in world_state.get("nearby_actors", []):
            if actor["distance"] < 25.0:
                self.llm_queried = True
                return True
                
        return False
