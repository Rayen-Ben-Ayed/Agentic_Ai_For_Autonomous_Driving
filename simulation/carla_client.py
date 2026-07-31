import logging
import os

import carla

logger = logging.getLogger(__name__)

# UE4 D3D11 occlusion queries often assert (DXGI_ERROR_INVALID_CALL) when the
# CARLA window keeps rendering while synchronous mode freezes the world during
# long LLM pauses. no_rendering_mode bypasses that path entirely.
_NO_RENDERING = os.getenv("CARLA_NO_RENDERING", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_FOLLOW_SPECTATOR = os.getenv("CARLA_FOLLOW_SPECTATOR", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# A good ego spawn has a long straight drivable lane ahead so scenarios can
# place obstacles directly in front of it.
_STRAIGHT_STEP_M = 2.0
_STRAIGHT_REQUIRED_M = 60.0
_STRAIGHT_MAX_LATERAL_M = 2.5


def _advance_same_lane(waypoint, step_m: float):
    successors = waypoint.next(step_m)
    if not successors:
        return None
    for nxt in successors:
        if nxt.road_id == waypoint.road_id and nxt.lane_id == waypoint.lane_id:
            return nxt
    return successors[0]


def _straight_run_lateral(carla_map, spawn_point) -> float | None:
    """Max lateral deviation (m) of this spawn's lane over the next
    `_STRAIGHT_REQUIRED_M`. Lower = straighter. None if the lane dead-ends."""
    wp = carla_map.get_waypoint(
        spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return None
    forward = spawn_point.get_forward_vector()
    right = spawn_point.get_right_vector()
    origin = spawn_point.location
    travelled = 0.0
    max_lateral = 0.0
    cur = wp
    while travelled < _STRAIGHT_REQUIRED_M:
        nxt = _advance_same_lane(cur, _STRAIGHT_STEP_M)
        if nxt is None:
            return None
        cur = nxt
        travelled += _STRAIGHT_STEP_M
        rel_x = cur.transform.location.x - origin.x
        rel_y = cur.transform.location.y - origin.y
        longitudinal = rel_x * forward.x + rel_y * forward.y
        lateral = rel_x * right.x + rel_y * right.y
        if longitudinal <= 0:
            return None
        max_lateral = max(max_lateral, abs(lateral))
    return max_lateral


def find_straight_spawn_point(world):
    """Pick the spawn point with the straightest lane ahead, so an obstacle can
    be placed directly in front of the ego. Falls back to None if none qualify."""
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    best_point = None
    best_lateral = None
    for sp in spawn_points:
        lateral = _straight_run_lateral(carla_map, sp)
        if lateral is None:
            continue
        if best_lateral is None or lateral < best_lateral:
            best_lateral = lateral
            best_point = sp
            if lateral <= _STRAIGHT_MAX_LATERAL_M:
                break
    if best_point is not None:
        logger.info(
            "Selected straight ego spawn (max lateral over %.0fm = %.2fm).",
            _STRAIGHT_REQUIRED_M,
            best_lateral,
        )
    return best_point


class CarlaClient:
    def __init__(self, host="127.0.0.1", port=2000, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self._original_settings = None
        self._synchronous = False

    def connect(self):
        try:
            logger.info("Connecting to CARLA server at %s:%s...", self.host, self.port)
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            logger.info("Successfully connected to CARLA server.")
        except Exception as e:
            logger.error("Failed to connect to CARLA server: %s", e)
            raise

    def enable_synchronous_mode(self, fixed_delta_seconds: float = 0.05):
        """Pin the simulator to a fixed time step driven by client ticks.

        In synchronous mode the world only advances when tick() is called, so
        slow agent/LLM decisions can no longer produce uncontrolled motion.
        """
        if not self.world:
            raise RuntimeError("World not initialized. Call connect() first.")
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        # Avoid UE render-thread occlusion crashes during LLM pauses.
        settings.no_rendering_mode = _NO_RENDERING
        self.world.apply_settings(settings)
        self._synchronous = True
        logger.info(
            "CARLA synchronous mode ON (fixed_delta_seconds=%.3f, no_rendering=%s)",
            fixed_delta_seconds,
            _NO_RENDERING,
        )

    def tick(self):
        """Advance one fixed-delta frame. No-op in asynchronous mode."""
        if self.world and self._synchronous:
            return self.world.tick()
        return None

    def is_synchronous(self) -> bool:
        return self._synchronous

    def _restore_settings(self):
        if self.world and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
                logger.info("CARLA simulation settings restored (async mode).")
            except Exception as e:
                logger.warning("Failed to restore CARLA settings: %s", e)
        self._synchronous = False
        self._original_settings = None

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
        ego_bp = blueprint_library.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "ego")

        candidates = [spawn_point] if spawn_point is not None else []
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points found in the map.")
            # Prefer a spawn with a long straight lane ahead so obstacles can be
            # placed directly in front; then fall back to any other spawn point.
            straight = find_straight_spawn_point(self.world)
            if straight is not None:
                candidates.append(straight)
            candidates.extend(spawn_points)

        self.ego_vehicle = None
        for candidate in candidates:
            self.ego_vehicle = self.world.try_spawn_actor(ego_bp, candidate)
            if self.ego_vehicle is not None:
                spawn_point = candidate
                break
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        logger.info("Ego vehicle spawned at %s", spawn_point.location)

        if _FOLLOW_SPECTATOR and not _NO_RENDERING:
            spectator = self.world.get_spectator()
            transform = carla.Transform(
                self.ego_vehicle.get_transform().location + carla.Location(z=50),
                carla.Rotation(pitch=-90),
            )
            spectator.set_transform(transform)

        return self.ego_vehicle

    @staticmethod
    def follow_spectator_enabled() -> bool:
        return _FOLLOW_SPECTATOR and not _NO_RENDERING

    def cleanup(self):
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
            logger.info("Ego vehicle destroyed.")
        self._restore_settings()
