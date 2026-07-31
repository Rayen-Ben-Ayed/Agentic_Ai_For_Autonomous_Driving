"""Scenario 8: three-lane blocked right + middle, clear leftmost overtake.

Layout (ego starts on the RIGHTMOST lane of a 3-lane road):
  - Right lane:  red stopped car ahead
  - Middle lane: second stopped car a bit further ahead than the red car
  - Leftmost:    empty — the only safe overtake path

Agent is expected to change left past the red car, discover the middle is also
blocked, then change left again onto the clear leftmost lane.

Placement constraint: world-state detection radius defaults to 65 m, and the
left-lane clear band ends at 45 m. The mid car must start inside 65 m (so it is
visible) but still far enough that the first left change past the red car remains
allowed.
"""
from __future__ import annotations

import logging
import os

import carla

from .base_scenario import BaseScenario
from .scenario_07_blocked_lane_clear_left import (
    _apply_vehicle_color,
    _ego_frame_offsets,
    _find_blueprint,
    _lane_is_clear_midroad,
    _waypoint_ahead_on_lane,
)

logger = logging.getLogger(__name__)

_NO_JUNCTION_M = 80.0
_BLOCKER_DIST_M = 42.0
# Keep mid inside DETECTION_RADIUS (~65 m): 42 + 16 = 58 m from ego at spawn.
_MID_AHEAD_OF_RED_M = 16.0
_DETECTION_RADIUS_M = float(os.getenv("DETECTION_RADIUS_M", "65.0"))


def _is_driving(wp) -> bool:
    return wp is not None and wp.lane_type == carla.LaneType.Driving


def _is_rightmost(wp) -> bool:
    right = wp.get_right_lane()
    return right is None or right.lane_type != carla.LaneType.Driving


def select_three_lane_rightmost_spawn(world):
    """Ego must sit on the rightmost of three consecutive driving lanes."""
    carla_map = world.get_map()
    candidates = []
    for index, spawn_point in enumerate(carla_map.get_spawn_points()):
        ego_wp = carla_map.get_waypoint(
            spawn_point.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None or ego_wp.is_junction or not _is_rightmost(ego_wp):
            continue
        mid_wp = ego_wp.get_left_lane()
        if not _is_driving(mid_wp):
            continue
        left_wp = mid_wp.get_left_lane()
        if not _is_driving(left_wp):
            continue
        if not _lane_is_clear_midroad(ego_wp, _NO_JUNCTION_M):
            continue
        if not _lane_is_clear_midroad(mid_wp, _NO_JUNCTION_M):
            continue
        if not _lane_is_clear_midroad(left_wp, _NO_JUNCTION_M):
            continue
        obstacle_wp, _ = _waypoint_ahead_on_lane(ego_wp, _BLOCKER_DIST_M)
        if obstacle_wp is None or obstacle_wp.is_junction:
            continue
        mid_at_blocker = obstacle_wp.get_left_lane()
        if not _is_driving(mid_at_blocker):
            continue
        candidates.append((index, spawn_point))

    if not candidates:
        logger.warning(
            "Scenario 08: no 3-lane rightmost spawn found; using CARLA default."
        )
        return None

    pick_index = min(len(candidates) - 1, len(candidates) // 3)
    original_index, spawn_point = candidates[pick_index]
    logger.info(
        "Scenario 08 selected 3-lane rightmost spawn #%d from %s (%d candidates).",
        original_index,
        world.get_map().name,
        len(candidates),
    )
    return spawn_point


def _hold_stopped(vehicle) -> None:
    vehicle.set_autopilot(False)
    vehicle.apply_control(
        carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
    )


def _spawn_stopped(world, bp, transform):
    """Spawn a stopped vehicle. Prefer try_spawn only (no force-spawn into collisions)."""
    for z_boost in (0.5, 0.8, 1.2):
        tf = carla.Transform(
            carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + z_boost,
            ),
            transform.rotation,
        )
        vehicle = world.try_spawn_actor(bp, tf)
        if vehicle is not None:
            _hold_stopped(vehicle)
            return vehicle
    return None


class Scenario08BlockedLaneUnsafeLeft(BaseScenario):
    def __init__(self, carla_client):
        super().__init__(carla_client)
        self.blocking_vehicle = None  # right-lane red stopper
        self.mid_lane_vehicle = None  # middle-lane stopper
        self.obstacle_distance_m = _BLOCKER_DIST_M
        self.llm_trigger_distance_m = 38.0
        self._last_logged_bucket = None

    def setup(self):
        ego = self.carla_client.get_ego_vehicle()
        if not ego:
            logger.error("Scenario 08: ego missing.")
            return

        carla_map = self.world.get_map()
        ego_tf = ego.get_transform()
        ego_wp = carla_map.get_waypoint(
            ego_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            logger.error("Scenario 08: no ego waypoint.")
            return

        mid_wp = ego_wp.get_left_lane()
        left_wp = mid_wp.get_left_lane() if _is_driving(mid_wp) else None
        logger.info(
            "Scenario 08 lane check: ego lane_id=%s rightmost=%s mid=%s leftmost=%s",
            ego_wp.lane_id,
            _is_rightmost(ego_wp),
            _is_driving(mid_wp),
            _is_driving(left_wp),
        )
        if not _is_driving(mid_wp) or not _is_driving(left_wp):
            logger.error(
                "Scenario 08: ego is NOT on a 3-lane rightmost segment. "
                "Mid car cannot be placed. Check spawn selection."
            )
            if not _is_driving(mid_wp):
                return

        library = self.world.get_blueprint_library()
        red_bp = _find_blueprint(
            library,
            [
                "vehicle.mercedes.coupe_2020",
                "vehicle.lincoln.mkz_2020",
                "vehicle.dodge.charger_2020",
                "vehicle.audi.tt",
            ],
        )
        _apply_vehicle_color(red_bp, (255, 0, 0))
        mid_bp = _find_blueprint(
            library,
            [
                "vehicle.audi.a2",
                "vehicle.toyota.prius",
                "vehicle.chevrolet.impala",
                "vehicle.tesla.model3",
            ],
        )
        _apply_vehicle_color(mid_bp, (30, 30, 30))

        right_spawn_wp = None
        for target_m in (self.obstacle_distance_m, 50.0, 40.0, 35.0):
            spawn_wp, actual_m = _waypoint_ahead_on_lane(ego_wp, target_m)
            if spawn_wp is None:
                continue
            lon, lat = _ego_frame_offsets(ego_tf, spawn_wp.transform.location)
            if lon <= 0.0 or abs(lat) > 3.0:
                continue
            vehicle = _spawn_stopped(self.world, red_bp, spawn_wp.transform)
            if vehicle is None:
                continue
            self.blocking_vehicle = vehicle
            self.npc_actors.append(vehicle)
            right_spawn_wp = spawn_wp
            logger.info(
                "Scenario 08: RED right-lane stopper at %.1fm (lane_id=%s).",
                actual_m,
                spawn_wp.lane_id,
            )
            break

        if self.blocking_vehicle is None:
            logger.error("Scenario 08: failed to spawn red right-lane stopper.")
            return

        self.carla_client.tick()

        if not self._spawn_mid_at_final_pose(ego_wp, right_spawn_wp, mid_bp):
            logger.error("Scenario 08: FAILED to spawn middle-lane stopped car.")
            return

        self.carla_client.tick()

        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        red_loc = self.blocking_vehicle.get_location()
        mid_loc = self.mid_lane_vehicle.get_location()
        red_lon, red_lat = _ego_frame_offsets(ego_tf, red_loc)
        mid_lon, mid_lat = _ego_frame_offsets(ego_tf, mid_loc)
        red_dist = ego_loc.distance(red_loc)
        mid_dist = ego_loc.distance(mid_loc)
        logger.info(
            "Scenario 08 FINAL: red lon=%.1fm lat=%.1fm dist=%.1fm | "
            "mid lon=%.1fm lat=%.1fm dist=%.1fm | mid_ahead_of_red=%.1fm | "
            "detection_radius=%.0fm",
            red_lon,
            red_lat,
            red_dist,
            mid_lon,
            mid_lat,
            mid_dist,
            mid_lon - red_lon,
            _DETECTION_RADIUS_M,
        )
        if mid_dist > _DETECTION_RADIUS_M:
            logger.error(
                "Scenario 08: mid car at %.1fm is OUTSIDE detection radius %.0fm — "
                "it will be invisible to the agent. Move it closer.",
                mid_dist,
                _DETECTION_RADIUS_M,
            )
        logger.info("=================================================")
        logger.info("SCENARIO 08: 3-LANE — RIGHT+MID BLOCKED, LEFT CLEAR")
        logger.info("Ego on rightmost. Overtake path = leftmost lane.")
        logger.info("=================================================")

    def _mid_final_waypoint(self, ego_wp, right_spawn_wp):
        """Middle-lane waypoint ~_MID_AHEAD_OF_RED_M ahead of the red car."""
        if right_spawn_wp is not None:
            mid_beside = right_spawn_wp.get_left_lane()
            if _is_driving(mid_beside):
                ahead, actual = _waypoint_ahead_on_lane(
                    mid_beside, _MID_AHEAD_OF_RED_M
                )
                if ahead is not None:
                    logger.info(
                        "Scenario 08: mid target = left-of-red + %.1fm.", actual
                    )
                    return ahead

        target_m = self.obstacle_distance_m + _MID_AHEAD_OF_RED_M
        ahead_wp, _ = _waypoint_ahead_on_lane(ego_wp, target_m)
        if ahead_wp is not None:
            mid = ahead_wp.get_left_lane()
            if _is_driving(mid):
                return mid

        mid_start = ego_wp.get_left_lane()
        if not _is_driving(mid_start):
            return None
        mid_ahead, _ = _waypoint_ahead_on_lane(mid_start, target_m)
        return mid_ahead

    def _spawn_mid_at_final_pose(self, ego_wp, right_spawn_wp, mid_bp) -> bool:
        """Spawn mid car once, at its final location — no set_transform / physics toggle."""
        mid_wp = self._mid_final_waypoint(ego_wp, right_spawn_wp)
        if mid_wp is None:
            logger.error("Scenario 08: could not resolve mid-lane final waypoint.")
            return False

        mid_vehicle = _spawn_stopped(self.world, mid_bp, mid_wp.transform)
        if mid_vehicle is None and right_spawn_wp is not None:
            beside = right_spawn_wp.get_left_lane()
            if _is_driving(beside):
                for delta in (2.0, -2.0, 4.0, 6.0):
                    dist = max(2.0, _MID_AHEAD_OF_RED_M + delta)
                    retry_wp, _ = _waypoint_ahead_on_lane(beside, dist)
                    if retry_wp is None:
                        continue
                    mid_vehicle = _spawn_stopped(
                        self.world, mid_bp, retry_wp.transform
                    )
                    if mid_vehicle is not None:
                        logger.info(
                            "Scenario 08: mid spawned on retry +%.0fm from red.",
                            dist,
                        )
                        break

        if mid_vehicle is None:
            return False

        self.mid_lane_vehicle = mid_vehicle
        self.npc_actors.append(mid_vehicle)
        logger.info(
            "Scenario 08: mid stopped car spawned at final pose (actor_id=%s).",
            mid_vehicle.id,
        )
        return True

    def update(self, step=None, *, allow_trigger=True):
        ego = self.carla_client.get_ego_vehicle()
        if self.blocking_vehicle and self.blocking_vehicle.is_alive:
            _hold_stopped(self.blocking_vehicle)
        if self.mid_lane_vehicle and self.mid_lane_vehicle.is_alive:
            _hold_stopped(self.mid_lane_vehicle)
        if not ego or not self.blocking_vehicle or not self.blocking_vehicle.is_alive:
            return
        distance = ego.get_location().distance(self.blocking_vehicle.get_location())
        bucket = int(distance // 5)
        if bucket != self._last_logged_bucket:
            self._last_logged_bucket = bucket
            logger.info("Scenario 08 | ego->red=%.1f m", distance)

    def is_llm_needed(self, world_state):
        if self.llm_queried:
            return False
        if world_state.get("path_blocked") and (
            world_state.get("effective_closest_distance") or 999.0
        ) < self.llm_trigger_distance_m:
            self.llm_queried = True
            return True
        return False
