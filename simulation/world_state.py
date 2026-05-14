import math

class WorldStateExtractor:
    def __init__(self, carla_client):
        self.carla_client = carla_client

    def get_state(self):
        """
        Extracts the current world state relevant for decision making.
        Returns a dictionary representing the state.
        """
        world = self.carla_client.get_world()
        ego_vehicle = self.carla_client.get_ego_vehicle()

        if not world or not ego_vehicle:
            return {"error": "World or ego vehicle not initialized"}

        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)

        # Get nearby actors (vehicles and pedestrians)
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.*')

        nearby_actors = []
        for actor in list(vehicles) + list(pedestrians):
            if actor.id == ego_vehicle.id:
                continue
            
            actor_transform = actor.get_transform()
            distance = ego_transform.location.distance(actor_transform.location)
            
            # Only consider actors within 50 meters
            if distance < 50.0:
                actor_velocity = actor.get_velocity()
                actor_speed = math.sqrt(actor_velocity.x**2 + actor_velocity.y**2 + actor_velocity.z**2)
                
                nearby_actors.append({
                    "id": actor.id,
                    "type": actor.type_id,
                    "distance": round(distance, 2),
                    "speed": round(actor_speed, 2),
                    "location": {
                        "x": round(actor_transform.location.x, 2),
                        "y": round(actor_transform.location.y, 2)
                    }
                })

        state = {
            "ego_vehicle": {
                "speed": round(ego_speed, 2),
                "location": {
                    "x": round(ego_transform.location.x, 2),
                    "y": round(ego_transform.location.y, 2)
                },
                "rotation": {
                    "yaw": round(ego_transform.rotation.yaw, 2)
                }
            },
            "nearby_actors": nearby_actors
        }

        return state
