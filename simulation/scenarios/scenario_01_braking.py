import carla
import logging

from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)

_NPC_DISTANCE_M = 50.0
_STEP_M = 2.0
# A candidate spawn must be genuinely ahead and within the ego lane, otherwise
# (on a curve/junction) it lands metres to the side and is never an obstacle.
_MAX_SPAWN_LATERAL_M = 3.0


def _ego_frame_offsets(ego_transform, location):
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    rel_x = location.x - ego_transform.location.x
    rel_y = location.y - ego_transform.location.y
    longitudinal = rel_x * forward.x + rel_y * forward.y
    lateral = rel_x * right.x + rel_y * right.y
    return longitudinal, lateral


def _advance_along_lane(waypoint, step_m: float):
    """Follow the ego's own lane by one short step.

    Prefers the successor that stays on the same road_id/lane_id so the path
    tracks the current lane through curves instead of jumping to a different
    lane at junctions (which `waypoint.next(50)` does on Town10)."""
    successors = waypoint.next(step_m)
    if not successors:
        return None
    for nxt in successors:
        if nxt.road_id == waypoint.road_id and nxt.lane_id == waypoint.lane_id:
            return nxt
    return successors[0]


def _waypoint_ahead_on_lane(ego_wp, target_distance_m: float):
    """Walk the lane ahead in small steps, returning a waypoint close to the
    target distance that is still on a driving lane. Returns (waypoint, dist)."""
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


class Scenario01Braking(BaseScenario):
    def setup(self):
        """Spawn a stopped Audi in the same driving lane, ~50 m ahead of ego."""
        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not found. Cannot setup scenario.")
            return

        ego_transform = ego_vehicle.get_transform()
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        blueprint_library = self.world.get_blueprint_library()
        npc_bp = blueprint_library.filter("vehicle.audi.tt")[0]
        npc_vehicle = None

        if ego_wp:
            # Prefer the farthest distance that is still straight ahead in-lane.
            for target_m in (50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0):
                spawn_wp, actual_m = _waypoint_ahead_on_lane(ego_wp, target_m)
                if spawn_wp is None:
                    continue
                longitudinal, lateral = _ego_frame_offsets(
                    ego_transform, spawn_wp.transform.location
                )
                if longitudinal <= 0 or abs(lateral) > _MAX_SPAWN_LATERAL_M:
                    logger.warning(
                        "Scenario 01: skip %.0fm ahead — off ego path "
                        "(lon=%.1fm lat=%.1fm, likely a curve).",
                        actual_m,
                        longitudinal,
                        lateral,
                    )
                    continue
                spawn_transform = spawn_wp.transform
                spawn_transform.location.z += 0.5
                candidate = self.world.try_spawn_actor(npc_bp, spawn_transform)
                if not candidate:
                    logger.warning(
                        "Scenario 01: spawn blocked at %.0fm ahead (occupied).", actual_m
                    )
                    continue
                npc_vehicle = candidate
                logger.info(
                    "Scenario 01 Setup: NPC %.0fm ahead in ego lane "
                    "(lane_id=%s, lat=%.1fm).",
                    actual_m,
                    spawn_wp.lane_id,
                    lateral,
                )
                break

        if npc_vehicle is None:
            logger.error(
                "Scenario 01 Setup: Failed to spawn NPC ahead on ego lane. "
                "Restart CARLA or move ego spawn."
            )
            return

        self.npc_actors.append(npc_vehicle)
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        npc_vehicle.apply_control(control)

    def is_llm_needed(self, world_state):
        if self.llm_queried:
            return False
        for actor in world_state.get("nearby_actors", []):
            if actor.get("is_scenario_npc") and actor.get("distance", 999) < 25.0:
                self.llm_queried = True
                return True
        return False
