import carla
import logging

logger = logging.getLogger(__name__)

class CarlaClient:
    def __init__(self, host='127.0.0.1', port=2000, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.ego_vehicle = None

    def connect(self):
        try:
            logger.info(f"Connecting to CARLA server at {self.host}:{self.port}...")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            logger.info("Successfully connected to CARLA server.")
        except Exception as e:
            logger.error(f"Failed to connect to CARLA server: {e}")
            raise

    def get_world(self):
        return self.world

    def set_ego_vehicle(self, vehicle):
        self.ego_vehicle = vehicle

    def get_ego_vehicle(self):
        return self.ego_vehicle

    def spawn_ego_vehicle(self, spawn_point=None):
        if not self.world:
            raise RuntimeError("World not initialized. Call connect() first.")
        
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')

        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points found in the map.")
            spawn_point = spawn_points[0]

        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_point)
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")
        
        logger.info(f"Ego vehicle spawned at {spawn_point.location}")

        # Set the spectator camera to follow the ego vehicle
        spectator = self.world.get_spectator()
        transform = carla.Transform(self.ego_vehicle.get_transform().location + carla.Location(z=50),
                                    carla.Rotation(pitch=-90))
        spectator.set_transform(transform)

        return self.ego_vehicle

    def cleanup(self):
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
            logger.info("Ego vehicle destroyed.")
