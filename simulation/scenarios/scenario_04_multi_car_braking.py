import carla
import logging

from simulation import lane_controller as lc
from simulation.junction_planner import classify_turn, straightest_waypoint

from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)

_STEP_M = 2.0
_MAX_SPAWN_LATERAL_M = 3.0
_MANEUVER_ZONE_MAX_M = 45.0
_LEFT_LOOKAHEAD_M = 6.0
_LEFT_ROUTE_ADVANCE_M = 3.5
_LEFT_ROUTE_REANCHOR_M = 10.0
_LEFT_LANE_MIN_CLEAR_M = 40.0


def _ego_frame_offsets(ego_transform, location):
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    rel_x = location.x - ego_transform.location.x
    rel_y = location.y - ego_transform.location.y
    longitudinal = rel_x * forward.x + rel_y * forward.y
    lateral = rel_x * right.x + rel_y * right.y
    return longitudinal, lateral


def _advance_along_lane(waypoint, step_m: float):
    successors = waypoint.next(step_m)
    if not successors:
        return None
    for nxt in successors:
        if nxt.road_id == waypoint.road_id and nxt.lane_id == waypoint.lane_id:
            return nxt
    return straightest_waypoint(successors, waypoint.transform.rotation.yaw) or successors[0]


def _waypoint_ahead_on_lane(ego_wp, target_distance_m: float):
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


def _waypoint_on_adjacent_lane(ego_wp, target_distance_m: float, side: str):
    ahead_wp, actual_m = _waypoint_ahead_on_lane(ego_wp, target_distance_m)
    if ahead_wp is None:
        return None, 0.0
    if side == "left":
        adj = ahead_wp.get_left_lane()
    else:
        adj = ahead_wp.get_right_lane()
    if adj is None or adj.lane_type != carla.LaneType.Driving:
        return None, 0.0
    return adj, actual_m


def _find_blueprint(blueprint_library, names):
    for name in names:
        try:
            return blueprint_library.find(name)
        except Exception:
            continue
    return list(blueprint_library.filter("vehicle.*"))[0]


def _spawn_vehicle(world, blueprint, waypoint):
    spawn_transform = waypoint.transform
    spawn_transform.location.z += 0.5
    vehicle = world.try_spawn_actor(blueprint, spawn_transform)
    if vehicle is None:
        return None
    vehicle.set_autopilot(False)
    return vehicle


def _drive_forward(vehicle, speed_mps: float, throttle: float = 0.5):
    forward = vehicle.get_transform().get_forward_vector()
    vehicle.set_target_velocity(
        carla.Vector3D(
            x=forward.x * speed_mps,
            y=forward.y * speed_mps,
            z=0.0,
        )
    )
    vehicle.apply_control(
        carla.VehicleControl(throttle=throttle, steer=0.0, brake=0.0, hand_brake=False)
    )


def _normalize_yaw_deg(yaw: float) -> float:
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return yaw


def _branch_is_short_dead_end(waypoint, max_m: float = 15.0) -> bool:
    """True when continuing straight ends within ``max_m`` with no further lane."""
    cur = waypoint
    travelled = 0.0
    while travelled < max_m:
        nxt = _advance_along_lane(cur, _STEP_M)
        if nxt is None:
            return not cur.next(_STEP_M)
        cur = nxt
        travelled += _STEP_M
    return False


def _advance_npc_along_lane(waypoint, step_m: float):
    """Follow the lane; at a split stay on the natural continuation of the road.

    Unlike ego junction planning (forward, then right, then left), NPCs keep
    the smallest heading change so they stay centered on their lane through
    curves and town-specific geometry.
    """
    nxt = _advance_along_lane(waypoint, step_m)
    if nxt is not None:
        return nxt

    candidates = waypoint.next(step_m)
    if not candidates:
        return waypoint

    entry_yaw = waypoint.transform.rotation.yaw
    straight = straightest_waypoint(candidates, entry_yaw)
    if straight is not None and not _branch_is_short_dead_end(straight):
        return straight

    turns = []
    for cand in candidates:
        if cand is straight:
            continue
        direction = classify_turn(entry_yaw, cand.transform.rotation.yaw)
        if direction == "u_turn":
            continue
        delta = abs(lc.normalize_yaw_error(cand.transform.rotation.yaw - entry_yaw))
        turns.append((delta, cand))
    if turns:
        return min(turns, key=lambda item: item[0])[1]
    return straight or candidates[0]


def _left_lane_spawn_usable(start_wp, min_clear_m: float = _LEFT_LANE_MIN_CLEAR_M) -> bool:
    """True when the left lane has enough drivable road ahead for the NPC."""
    wp = start_wp
    travelled = 0.0
    while travelled < min_clear_m:
        if wp.lane_type != carla.LaneType.Driving:
            return False
        nxt = _advance_along_lane(wp, _STEP_M)
        if nxt is None:
            return False
        wp = nxt
        travelled += _STEP_M
    return True


def _waypoint_lookahead_from(start_wp, distance_m: float):
    wp = start_wp
    travelled = 0.0
    while travelled < distance_m:
        step = min(_STEP_M, distance_m - travelled)
        nxt = _advance_npc_along_lane(wp, step)
        if nxt is wp:
            break
        wp = nxt
        travelled += step
    return wp


def _drive_along_waypoint(vehicle, target_wp, speed_mps: float, throttle: float = 0.3):
    veh_tf = vehicle.get_transform()
    target_tf = target_wp.transform
    forward = veh_tf.get_forward_vector()
    right = veh_tf.get_right_vector()
    rel = target_tf.location - veh_tf.location
    lateral = rel.x * right.x + rel.y * right.y
    yaw_err = _normalize_yaw_deg(target_tf.rotation.yaw - veh_tf.rotation.yaw)
    steer = lc.compute_steer(
        lateral,
        yaw_err,
        lat_gain=lc.LAT_GAIN,
        yaw_gain=lc.YAW_GAIN,
        max_steer=lc.MAX_STEER_FOLLOW,
    )

    vehicle.set_target_velocity(
        carla.Vector3D(
            x=forward.x * speed_mps,
            y=forward.y * speed_mps,
            z=0.0,
        )
    )
    vehicle.apply_control(
        carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0, hand_brake=False)
    )


class Scenario04MultiCarBraking(BaseScenario):
    """Multi-car traffic in three lanes; ego lane lead + moving side traffic.

    Layout (typical):
      - Slow lead in the ego lane ~30 m ahead (~2.0 m/s).
      - Red vehicle in the right lane ~42 m ahead (~1.3 m/s).
      - Slow vehicle in the left lane ~72 m ahead (~2.5 m/s); follows the map
        and turns at the road end instead of driving straight into a barrier.
    """

    def __init__(self, carla_client):
        super().__init__(carla_client)
        self.primary_npc = None
        self.right_npc = None
        self.left_npc = None
        self.llm_trigger_distance_m = 30.0
        self.primary_distance_m = 30.0
        self.right_distance_m = 42.0
        self.left_distance_m = 72.0
        # Ego cruise cap is ~8 m/s — NPCs must stay well below that to be catchable.
        self.primary_speed_mps = 2.0
        self.right_speed_mps = 1.3
        self.left_speed_mps = 2.5
        self._left_route_wp = None

    def setup(self):
        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not found. Cannot setup scenario.")
            return

        ego_transform = ego_vehicle.get_transform()
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_wp is None:
            logger.error("Scenario 04: no ego lane waypoint.")
            return

        blueprint_library = self.world.get_blueprint_library()
        primary_bp = _find_blueprint(blueprint_library, ["vehicle.audi.tt"])
        right_bp = _find_blueprint(
            blueprint_library,
            ["vehicle.dodge.charger_2020", "vehicle.tesla.model3", "vehicle.audi.a2"],
        )
        if right_bp.has_attribute("color"):
            right_bp.set_attribute("color", "255,0,0")
        left_bp = _find_blueprint(
            blueprint_library,
            ["vehicle.mercedes.coupe_2020", "vehicle.lincoln.mkz_2020", "vehicle.bmw.grandtourer"],
        )

        left_available = ego_wp.get_left_lane()
        left_available = (
            left_available is not None and left_available.lane_type == carla.LaneType.Driving
        )
        right_available = ego_wp.get_right_lane()
        right_available = (
            right_available is not None and right_available.lane_type == carla.LaneType.Driving
        )

        if left_available:
            escape_side = "left"
            block_side = "right" if right_available else None
        elif right_available:
            escape_side = "right"
            block_side = None
        else:
            escape_side = None
            block_side = None
            logger.warning(
                "Scenario 04: single-lane road — only the ego-lane lead will spawn."
            )

        spawned = 0

        for target_m in (
            self.primary_distance_m,
            28.0,
            32.0,
            26.0,
            35.0,
        ):
            spawn_wp, actual_m = _waypoint_ahead_on_lane(ego_wp, target_m)
            if spawn_wp is None:
                continue
            longitudinal, lateral = _ego_frame_offsets(
                ego_transform, spawn_wp.transform.location
            )
            if longitudinal <= 0 or abs(lateral) > _MAX_SPAWN_LATERAL_M:
                logger.warning(
                    "Scenario 04: skip %.0fm ahead in ego lane (lon=%.1fm lat=%.1fm).",
                    actual_m,
                    longitudinal,
                    lateral,
                )
                continue
            self.primary_npc = _spawn_vehicle(self.world, primary_bp, spawn_wp)
            if self.primary_npc is None:
                logger.warning("Scenario 04: ego-lane spawn blocked at %.0fm.", actual_m)
                continue
            self.npc_actors.append(self.primary_npc)
            spawned += 1
            logger.info(
                "Scenario 04: primary NPC %.0fm ahead in ego lane (lane_id=%s).",
                actual_m,
                spawn_wp.lane_id,
            )
            break

        if self.primary_npc is None:
            logger.error(
                "Scenario 04: failed to spawn primary NPC on ego lane. "
                "Restart CARLA or move ego spawn."
            )
            return

        if block_side == "right":
            for target_m in (
                self.right_distance_m,
                40.0,
                38.0,
                44.0,
                36.0,
            ):
                side_wp, actual_m = _waypoint_on_adjacent_lane(
                    ego_wp, target_m, "right"
                )
                if side_wp is None:
                    continue
                longitudinal, lateral = _ego_frame_offsets(
                    ego_transform, side_wp.transform.location
                )
                if longitudinal <= 0 or abs(lateral) > 5.5:
                    continue
                right_vehicle = _spawn_vehicle(self.world, right_bp, side_wp)
                if right_vehicle is None:
                    continue
                self.right_npc = right_vehicle
                self.npc_actors.append(right_vehicle)
                spawned += 1
                logger.info(
                    "Scenario 04: red right-lane vehicle %.0fm ahead (lane_id=%s).",
                    actual_m,
                    side_wp.lane_id,
                )
                break

        if escape_side == "left":
            for target_m in (
                self.left_distance_m,
                70.0,
                74.0,
                68.0,
                76.0,
            ):
                side_wp, actual_m = _waypoint_on_adjacent_lane(
                    ego_wp, target_m, "left"
                )
                if side_wp is None:
                    continue
                if not _left_lane_spawn_usable(side_wp):
                    logger.warning(
                        "Scenario 04: skip %.0fm left-lane spawn (insufficient clear road).",
                        actual_m,
                    )
                    continue
                longitudinal, _ = _ego_frame_offsets(
                    ego_transform, side_wp.transform.location
                )
                if longitudinal <= _MANEUVER_ZONE_MAX_M:
                    continue
                left_vehicle = _spawn_vehicle(self.world, left_bp, side_wp)
                if left_vehicle is None:
                    continue
                self.left_npc = left_vehicle
                self._left_route_wp = side_wp
                self.npc_actors.append(left_vehicle)
                spawned += 1
                logger.info(
                    "Scenario 04: left-lane traffic %.0fm ahead (lane_id=%s).",
                    actual_m,
                    side_wp.lane_id,
                )
                break

        for actor in self.npc_actors:
            self._drive_npc(actor)

        logger.info("=================================================")
        logger.info("SCENARIO 04: MULTI-CAR TRAFFIC (NAVIGABLE)")
        logger.info("Spawned %d moving NPC(s).", spawned)
        logger.info(
            "Ego lead ~%.0fm @ %.1f m/s | red right ~%.0fm @ %.1f m/s | "
            "left ~%.0fm @ %.1f m/s",
            self.primary_distance_m,
            self.primary_speed_mps,
            self.right_distance_m,
            self.right_speed_mps,
            self.left_distance_m,
            self.left_speed_mps,
        )
        if escape_side:
            logger.info(
                "Escape route: change_lane_%s when safe.",
                escape_side,
            )
        logger.info("=================================================")

    def _drive_npc(self, vehicle):
        if vehicle is self.primary_npc:
            _drive_forward(vehicle, self.primary_speed_mps, throttle=0.28)
        elif vehicle is self.right_npc:
            _drive_forward(vehicle, self.right_speed_mps, throttle=0.18)
        elif vehicle is self.left_npc:
            self._drive_left_npc(vehicle)

    def _drive_left_npc(self, vehicle):
        carla_map = self.world.get_map()
        veh_loc = vehicle.get_transform().location
        lane_wp = carla_map.get_waypoint(
            veh_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if lane_wp is None:
            _drive_forward(vehicle, self.left_speed_mps, throttle=0.28)
            return

        if self._left_route_wp is None:
            self._left_route_wp = lane_wp
        elif veh_loc.distance(self._left_route_wp.transform.location) > _LEFT_ROUTE_REANCHOR_M:
            self._left_route_wp = lane_wp

        route_loc = self._left_route_wp.transform.location
        if veh_loc.distance(route_loc) < _LEFT_ROUTE_ADVANCE_M:
            advanced = _advance_npc_along_lane(self._left_route_wp, _STEP_M)
            if advanced is not None:
                self._left_route_wp = advanced

        lookahead = _waypoint_lookahead_from(self._left_route_wp, _LEFT_LOOKAHEAD_M)
        _drive_along_waypoint(vehicle, lookahead, self.left_speed_mps, throttle=0.28)

    def update(self, step=None, *, allow_trigger=True):
        for actor in self.npc_actors:
            if actor.is_alive:
                self._drive_npc(actor)

    def is_llm_needed(self, world_state):
        if self.llm_queried:
            return False
        for actor in world_state.get("nearby_actors", []):
            if actor.get("is_scenario_npc") and actor.get("distance", 999) < self.llm_trigger_distance_m:
                self.llm_queried = True
                return True
        return False

    def teardown(self):
        for actor in self.npc_actors:
            if actor.is_alive:
                try:
                    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception as exc:
                    logger.warning("Scenario 04: could not stop NPC: %s", exc)
        super().teardown()
