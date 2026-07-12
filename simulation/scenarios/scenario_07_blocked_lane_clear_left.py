import logging

import carla

from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)

_STEP_M = 2.0
_MAX_LATERAL_M = 3.0
_CLEAR_ROAD_M = 90.0
_NO_JUNCTION_M = 80.0


SCENARIO_MAPS = {
    "6": "Town02",
    "7": "Town04",
    "8": "Town05",
}


def _ego_frame_offsets(ego_transform, location):
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    rel_x = location.x - ego_transform.location.x
    rel_y = location.y - ego_transform.location.y
    longitudinal = rel_x * forward.x + rel_y * forward.y
    lateral = rel_x * right.x + rel_y * right.y
    return longitudinal, lateral


def _advance_along_lane(waypoint, step_m):
    successors = waypoint.next(step_m)
    if not successors:
        return None
    for nxt in successors:
        if nxt.road_id == waypoint.road_id and nxt.lane_id == waypoint.lane_id:
            return nxt
    return successors[0]


def _waypoint_ahead_on_lane(ego_wp, target_distance_m):
    wp = ego_wp
    travelled = 0.0
    while travelled < target_distance_m:
        nxt = _advance_along_lane(wp, _STEP_M)
        if nxt is None:
            break
        wp = nxt
        travelled += _STEP_M
    if travelled <= 0.0 or wp.lane_type != carla.LaneType.Driving:
        return None, 0.0
    return wp, travelled


def _find_blueprint(blueprint_library, names):
    for name in names:
        try:
            return blueprint_library.find(name)
        except Exception:
            continue
    return list(blueprint_library.filter("vehicle.*"))[0]


def _lane_is_clear_midroad(waypoint, distance_m):
    wp = waypoint
    travelled = 0.0
    while travelled < distance_m:
        if wp.is_junction:
            return False
        nxt = _advance_along_lane(wp, _STEP_M)
        if nxt is None or nxt.lane_type != carla.LaneType.Driving:
            return False
        wp = nxt
        travelled += _STEP_M
    return True


def select_midroad_spawn_point(world, variant="7"):
    """Pick a realistic non-junction road segment for scenarios 7/8.

    The selected spawn must have a left driving lane and a long clear corridor
    ahead, so the obstacle is placed in the middle of a road rather than inside
    a crossing or intersection.
    """
    carla_map = world.get_map()
    candidates = []
    for index, spawn_point in enumerate(carla_map.get_spawn_points()):
        ego_wp = carla_map.get_waypoint(
            spawn_point.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None or ego_wp.is_junction:
            continue
        left_wp = ego_wp.get_left_lane()
        if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
            continue
        if not _lane_is_clear_midroad(ego_wp, _NO_JUNCTION_M):
            continue
        if not _lane_is_clear_midroad(left_wp, _NO_JUNCTION_M):
            continue

        obstacle_wp, _ = _waypoint_ahead_on_lane(ego_wp, 55.0)
        if obstacle_wp is None or obstacle_wp.is_junction:
            continue
        candidates.append((index, spawn_point))

    if not candidates:
        logger.warning(
            "No ideal mid-road spawn found for scenario %s; using CARLA default spawn.",
            variant,
        )
        return None

    # Use different valid segments for 7 and 8 so the demos do not look cloned.
    pick_index = 0 if variant == "7" else min(len(candidates) - 1, len(candidates) // 2)
    original_index, spawn_point = candidates[pick_index]
    logger.info(
        "Scenario %s selected mid-road ego spawn #%d from %s.",
        variant,
        original_index,
        world.get_map().name,
    )
    return spawn_point


class Scenario07BlockedLaneClearLeft(BaseScenario):
    def __init__(self, carla_client):
        super().__init__(carla_client)
        self.blocking_vehicle = None
        self.control_ego = False
        self.ego_throttle = 0.55
        self.obstacle_distance_m = 55.0
        self.llm_trigger_distance_m = 38.0
        self._last_logged_bucket = None

    def setup(self):
        """Stopped vehicle in ego lane; left adjacent lane is intended to be clear."""
        ego = self.carla_client.get_ego_vehicle()
        if not ego:
            logger.error("Ego vehicle not found. Cannot setup scenario 07.")
            return

        ego_tf = ego.get_transform()
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            ego_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            logger.error("Scenario 07: no ego lane waypoint.")
            return

        left_lane = ego_wp.get_left_lane()
        if left_lane is None or left_lane.lane_type != carla.LaneType.Driving:
            logger.warning(
                "Scenario 07: no driving lane on the left at ego start; "
                "agent may need to yield/stop instead of changing lane."
            )

        blueprint_library = self.world.get_blueprint_library()
        bp = _find_blueprint(
            blueprint_library,
            [
                "vehicle.mercedes.coupe_2020",
                "vehicle.lincoln.mkz_2020",
                "vehicle.dodge.charger_2020",
                "vehicle.audi.tt",
            ],
        )

        for target_m in (self.obstacle_distance_m, 50.0, 45.0, 40.0):
            spawn_wp, actual_m = _waypoint_ahead_on_lane(ego_wp, target_m)
            if spawn_wp is None:
                continue
            longitudinal, lateral = _ego_frame_offsets(ego_tf, spawn_wp.transform.location)
            if longitudinal <= 0.0 or abs(lateral) > _MAX_LATERAL_M:
                continue
            spawn_transform = spawn_wp.transform
            spawn_transform.location.z += 0.5
            self.blocking_vehicle = self.world.try_spawn_actor(bp, spawn_transform)
            if self.blocking_vehicle is not None:
                logger.info(
                    "Scenario 07: blocker spawned %.1fm ahead in ego lane.",
                    actual_m,
                )
                break

        if self.blocking_vehicle is None:
            logger.error("Scenario 07: failed to spawn the blocking vehicle.")
            return

        self.npc_actors.append(self.blocking_vehicle)
        self.blocking_vehicle.set_autopilot(False)
        self.blocking_vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )

        self.world.debug.draw_string(
            self.blocking_vehicle.get_location() + carla.Location(z=3.0),
            "BLOCKED LANE - LEFT CLEAR",
            color=carla.Color(255, 0, 0),
            life_time=30.0,
        )
        logger.info("=================================================")
        logger.info("SCENARIO 07: BLOCKED LANE, LEFT LANE CLEAR")
        logger.info("No-agent: ego drives into the stopped vehicle.")
        logger.info("Agent: expected response is change_lane_left if safe.")
        logger.info("=================================================")

    def update(self, step=None, *, allow_trigger=True):
        ego = self.carla_client.get_ego_vehicle()
        if not ego or not self.blocking_vehicle or not self.blocking_vehicle.is_alive:
            return

        if self.control_ego:
            ego.apply_control(
                carla.VehicleControl(throttle=self.ego_throttle, steer=0.0, brake=0.0)
            )

        self.blocking_vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )

        distance = ego.get_location().distance(self.blocking_vehicle.get_location())
        bucket = int(distance // 5)
        if bucket != self._last_logged_bucket:
            self._last_logged_bucket = bucket
            logger.info("Scenario 07 | ego->blocker=%.1f m", distance)

    def is_llm_needed(self, world_state):
        if self.llm_queried:
            return False
        if world_state.get("path_blocked") and (
            world_state.get("effective_closest_distance") or 999.0
        ) < self.llm_trigger_distance_m:
            self.llm_queried = True
            logger.info("Scenario 07: blocker detected, LLM should decide now.")
            return True
        return False
