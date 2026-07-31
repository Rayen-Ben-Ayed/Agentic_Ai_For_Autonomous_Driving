"""Scenario 6: left-lane vehicle merges into ego lane on Town10.

A second vehicle cruises in the left lane ahead of ego, then merges into the
ego lane when ego enters the trigger range. Uses Town10HD_Opt with the same
straight spawn strategy as scenario 5.
"""
from __future__ import annotations

import logging

import carla

from simulation import lane_controller as lc
from simulation.carla_client import find_straight_spawn_point
from simulation.timing_config import CARLA_FIXED_DELTA_S

from .base_scenario import BaseScenario
from .scenario_04_multi_car_braking import (
    _advance_npc_along_lane,
    _drive_along_waypoint,
    _find_blueprint,
    _waypoint_ahead_on_lane,
    _waypoint_lookahead_from,
)
from .scenario_07_blocked_lane_clear_left import _advance_along_lane

logger = logging.getLogger(__name__)

_STEP_M = 2.0
_CLEAR_ROAD_M = 85.0
_NPC_START_M = 46.0
_MERGE_POINT_M = 68.0
_MIN_MERGE_TRAVEL_M = 55.0
_MAX_SEGMENT_YAW_DEG = 6.0
_ROADSIDE_OFFSET_M = 3.2
_ROUTE_LOOKAHEAD_M = 6.0
_LANE_ROUTE_ADVANCE_M = 3.5
_LANE_ROUTE_REANCHOR_M = 10.0
_MERGE_OFFSET_STEP_M = 0.03
_MERGE_OFFSET_LEAD_M = 0.35


def _normalize_yaw_deg(yaw: float) -> float:
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return yaw


def _segment_max_yaw_change(start_wp, distance_m: float, step_m: float = _STEP_M):
    wp = start_wp
    travelled = 0.0
    prev_yaw = wp.transform.rotation.yaw
    max_delta = 0.0
    while travelled < distance_m:
        nxt = _advance_along_lane(wp, step_m)
        if nxt is None:
            break
        yaw = nxt.transform.rotation.yaw
        max_delta = max(max_delta, abs(_normalize_yaw_deg(yaw - prev_yaw)))
        prev_yaw = yaw
        wp = nxt
        travelled += step_m
    return max_delta, travelled, wp


def _segment_is_straight(start_wp, distance_m: float, max_yaw_deg: float = _MAX_SEGMENT_YAW_DEG):
    max_delta, travelled, _ = _segment_max_yaw_change(start_wp, distance_m)
    if travelled < distance_m - 6.0:
        return False
    return max_delta <= max_yaw_deg


def _find_merge_waypoint(ego_wp, preferred_distances):
    for distance_m in preferred_distances:
        merge_wp, travelled = _waypoint_ahead_on_lane(ego_wp, distance_m)
        if merge_wp is None or merge_wp.is_junction:
            continue
        if travelled < _MIN_MERGE_TRAVEL_M:
            continue
        if _segment_is_straight(ego_wp, travelled):
            return merge_wp, travelled
    for distance_m in preferred_distances:
        merge_wp, travelled = _waypoint_ahead_on_lane(ego_wp, distance_m)
        if merge_wp is None or merge_wp.is_junction:
            continue
        if travelled >= 40.0:
            return merge_wp, travelled
    return None, 0.0


def _find_npc_start_waypoint(ego_wp, preferred_distances):
    for distance_m in preferred_distances:
        start_wp, travelled = _waypoint_ahead_on_lane(ego_wp, distance_m)
        if start_wp is None or start_wp.is_junction:
            continue
        if travelled < distance_m - 8.0:
            continue
        if _segment_is_straight(start_wp, min(20.0, _CLEAR_ROAD_M - travelled)):
            return start_wp, travelled
    for distance_m in preferred_distances:
        start_wp, travelled = _waypoint_ahead_on_lane(ego_wp, distance_m)
        if start_wp is None or start_wp.is_junction:
            continue
        if travelled >= 25.0:
            return start_wp, travelled
    return None, 0.0


def select_right_lane_pullout_spawn_point(world):
    """Use the same straight Town10 spawn that scenario 5 relies on."""
    straight = find_straight_spawn_point(world)
    if straight is not None:
        logger.info(
            "Scenario 06 using straight ego spawn from %s (scenario 5 strategy).",
            world.get_map().name,
        )
        return straight

    logger.warning("Scenario 06: no straight spawn found; using CARLA default.")
    return None


class Scenario06RightLanePullout(BaseScenario):
    """Left-lane vehicle merges into ego lane on Town10."""

    def __init__(self, carla_client):
        super().__init__(carla_client)
        self.pullout_vehicle = None
        self.control_ego = False
        self.ego_throttle = 0.50
        self.npc_start_distance_m = _NPC_START_M
        self.merge_target_distance_m = _MERGE_POINT_M
        self.trigger_distance_m = 58.0
        self.left_lane_speed_mps = 2.8
        self.ego_lane_speed_mps = 4.5
        self.merge_started = False
        self.merge_complete = False
        self.merge_target_location = None
        self._source_route_wp = None
        self._ego_lane_route_wp = None
        self._ego_lane_id = None
        self.lateral_offset_m = _ROADSIDE_OFFSET_M
        self._merge_stable_ticks = 0
        self._last_logged_bucket = None

    def setup(self):
        ego = self.carla_client.get_ego_vehicle()
        if not ego:
            logger.error("Ego vehicle not found. Cannot setup scenario 06.")
            return

        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            ego.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            logger.error("Scenario 06: no ego lane waypoint.")
            return

        self._ego_lane_id = ego_wp.lane_id

        merge_wp, merge_travel = _find_merge_waypoint(
            ego_wp,
            (
                self.merge_target_distance_m,
                74.0,
                70.0,
                68.0,
                62.0,
                58.0,
                52.0,
                _MIN_MERGE_TRAVEL_M,
            ),
        )
        npc_start_wp, npc_travel = _find_npc_start_waypoint(
            ego_wp,
            (
                self.npc_start_distance_m,
                44.0,
                48.0,
                42.0,
                50.0,
                38.0,
                34.0,
                30.0,
            ),
        )
        if merge_wp is None or npc_start_wp is None:
            logger.error("Scenario 06: failed to find pullout waypoints on this map segment.")
            return

        self.merge_target_distance_m = merge_travel
        if merge_travel < 20.0 or npc_travel < 15.0:
            logger.error(
                "Scenario 06: pullout geometry too short (merge=%.1fm, npc=%.1fm).",
                merge_travel,
                npc_travel,
            )
            return

        self.merge_target_location = merge_wp.transform.location
        self.merge_target_location.z += 0.5
        ego_lane_start_wp, _ = _waypoint_ahead_on_lane(ego_wp, npc_travel)
        if ego_lane_start_wp is None:
            logger.error("Scenario 06: failed to resolve ego-lane merge path.")
            return
        self._ego_lane_route_wp = ego_lane_start_wp

        blueprint_library = self.world.get_blueprint_library()
        bp = _find_blueprint(
            blueprint_library,
            [
                "vehicle.dodge.charger_2020",
                "vehicle.tesla.model3",
                "vehicle.audi.a2",
            ],
        )
        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")

        self.pullout_vehicle = self._spawn_on_roadside(bp, npc_start_wp)
        if self.pullout_vehicle is None:
            logger.error("Scenario 06: failed to spawn pullout vehicle.")
            return
        if self._source_route_wp is None:
            left_wp = npc_start_wp.get_left_lane()
            self._source_route_wp = (
                left_wp
                if left_wp is not None and left_wp.lane_type == carla.LaneType.Driving
                else npc_start_wp
            )

        self.lateral_offset_m = max(
            _ROADSIDE_OFFSET_M,
            self._lateral_separation_from_ego_lane_m(),
        )

        self.npc_actors.append(self.pullout_vehicle)
        self.pullout_vehicle.set_autopilot(False)
        self._drive_source_lane_npc(self.left_lane_speed_mps)

        logger.info("=================================================")
        logger.info("SCENARIO 06: RED CAR MERGES FROM LEFT LANE")
        logger.info(
            "Red car cruises in the left lane %.1fm ahead @ %.1f m/s.",
            npc_travel,
            self.left_lane_speed_mps,
        )
        logger.info(
            "It merges into ego lane %.1fm ahead @ %.1f m/s when ego is within %.1fm.",
            self.merge_target_distance_m,
            self.ego_lane_speed_mps,
            self.trigger_distance_m,
        )
        logger.info("No-agent: ego continues into the merging vehicle.")
        logger.info("Agent: expected response is yield/stop or safe lane change.")
        logger.info("=================================================")

    def update(self, step=None, *, allow_trigger=True):
        ego = self.carla_client.get_ego_vehicle()
        if not ego or not self.pullout_vehicle or not self.pullout_vehicle.is_alive:
            return

        if self.control_ego:
            ego.apply_control(
                carla.VehicleControl(throttle=self.ego_throttle, steer=0.0, brake=0.0)
            )

        ego_distance = ego.get_location().distance(self.merge_target_location)
        npc_distance = self.pullout_vehicle.get_location().distance(self.merge_target_location)

        if not self.merge_started:
            self._drive_source_lane_npc(self.left_lane_speed_mps)
            self._ego_lane_route_wp = self._advance_route_wp(
                self._ego_lane_route_wp,
                self.left_lane_speed_mps * CARLA_FIXED_DELTA_S,
            )
            should_trigger = allow_trigger and (
                ego_distance < self.trigger_distance_m
                or npc_distance < 18.0
            )
            if should_trigger:
                self.merge_started = True
                self._merge_stable_ticks = 0
                self._sync_merge_route_to_npc()
                self.lateral_offset_m = max(
                    0.8,
                    min(4.2, self._lateral_separation_from_ego_lane_m()),
                )
                logger.info(
                    "Scenario 06: merge starts now (ego->conflict=%.1fm, npc->conflict=%.1fm, lat_to_ego=%.1fm, offset=%.1fm).",
                    ego_distance,
                    npc_distance,
                    self._lateral_separation_from_ego_lane_m(),
                    self.lateral_offset_m,
                )
        elif not self.merge_complete:
            self._update_merge_lateral_offset()
            self._advance_merge_route(self.ego_lane_speed_mps * CARLA_FIXED_DELTA_S)
            current_sep = self._lateral_separation_from_ego_lane_m()
            throttle = 0.55 if current_sep > 0.8 else 0.38
            self._drive_along_merge_route(
                self.ego_lane_speed_mps,
                throttle=throttle,
                lane_change=True,
            )

            lat_err = current_sep
            if lat_err < lc.CENTER_TOLERANCE_M:
                self._merge_stable_ticks += 1
            else:
                self._merge_stable_ticks = 0
            if self._merge_stable_ticks >= 20:
                self.merge_complete = True
                logger.info(
                    "Scenario 06: merge complete on ego lane (lateral error=%.2fm, offset=%.2fm).",
                    lat_err,
                    self.lateral_offset_m,
                )
        else:
            self._advance_merge_route(self.ego_lane_speed_mps * CARLA_FIXED_DELTA_S)
            self._drive_along_merge_route(self.ego_lane_speed_mps, throttle=0.32, lane_change=False)

        bucket = int(ego_distance // 5)
        if bucket != self._last_logged_bucket:
            self._last_logged_bucket = bucket
            logger.info(
                "Scenario 06 | ego->conflict=%.1f m | npc->conflict=%.1f m | phase=%s | lat_to_ego=%.1fm | offset=%.1fm",
                ego_distance,
                npc_distance,
                "merged" if self.merge_complete else ("merging" if self.merge_started else "cruise"),
                self._lateral_separation_from_ego_lane_m(),
                self.lateral_offset_m,
            )

    def _signed_lateral_to_ego_lane_m(self) -> float:
        """Signed lateral offset vs ego lane (+ = right of ego lane center)."""
        veh_tf = self.pullout_vehicle.get_transform()
        ego_lane_wp = self._ego_lane_wp_at(veh_tf.location)
        if ego_lane_wp is None:
            return -self.lateral_offset_m
        right = ego_lane_wp.transform.get_right_vector()
        return lc.lateral_error_m(
            veh_tf.location.x,
            veh_tf.location.y,
            ego_lane_wp.transform.location.x,
            ego_lane_wp.transform.location.y,
            right.x,
            right.y,
        )

    def _lateral_separation_from_ego_lane_m(self) -> float:
        """Distance left of ego lane center while still in the left lane."""
        signed = self._signed_lateral_to_ego_lane_m()
        if signed >= 0.0:
            return signed
        return -signed

    def _update_merge_lateral_offset(self):
        """Shrink merge target only as fast as the car can follow it."""
        current_sep = self._lateral_separation_from_ego_lane_m()
        proposed = max(0.0, self.lateral_offset_m - _MERGE_OFFSET_STEP_M)
        if current_sep > 0.5:
            floor = max(0.0, current_sep - _MERGE_OFFSET_LEAD_M)
            self.lateral_offset_m = max(proposed, floor)
        else:
            self.lateral_offset_m = proposed

    def is_llm_needed(self, world_state):
        if self.llm_queried:
            return False
        for actor in world_state.get("nearby_actors", []):
            if actor.get("is_scenario_npc") and actor.get("distance", 999.0) < 42.0:
                self.llm_queried = True
                logger.info(
                    "Scenario 06: pullout vehicle detected at %.1fm, LLM should decide.",
                    actor.get("distance"),
                )
                return True
        return False

    def _ego_lane_wp_at(self, location):
        wp = self.world.get_map().get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None or self._ego_lane_id is None:
            return None
        if wp.lane_id == self._ego_lane_id:
            return wp
        for _ in range(3):
            right = wp.get_right_lane()
            if right is None or right.lane_type != carla.LaneType.Driving:
                return None
            wp = right
            if wp.lane_id == self._ego_lane_id:
                return wp
        return None

    def _sync_merge_route_to_npc(self):
        ego_lane_wp = self._ego_lane_wp_at(self.pullout_vehicle.get_transform().location)
        if ego_lane_wp is not None:
            self._ego_lane_route_wp = ego_lane_wp

    def _reanchor_route_wp(self, route_wp):
        carla_map = self.world.get_map()
        lane_wp = carla_map.get_waypoint(
            self.pullout_vehicle.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if lane_wp is None:
            return route_wp
        if route_wp is None:
            return lane_wp
        veh_loc = self.pullout_vehicle.get_transform().location
        if veh_loc.distance(route_wp.transform.location) > _LANE_ROUTE_REANCHOR_M:
            synced = self._ego_lane_wp_at(veh_loc)
            return synced if synced is not None else route_wp
        return route_wp

    def _advance_route_wp(self, route_wp, distance_m: float):
        if route_wp is None or distance_m <= 0.0:
            return route_wp
        remaining = distance_m
        wp = route_wp
        while remaining > 0.05:
            step_m = min(_STEP_M, remaining)
            nxt = _advance_npc_along_lane(wp, step_m)
            if nxt is wp:
                break
            wp = nxt
            remaining -= step_m
        return wp

    def _advance_merge_route(self, distance_m: float):
        if self._ego_lane_route_wp is None or distance_m <= 0.0:
            return
        remaining = distance_m
        while remaining > 0.05:
            step_m = min(_STEP_M, remaining)
            nxt = _advance_along_lane(self._ego_lane_route_wp, step_m)
            if nxt is None:
                break
            self._ego_lane_route_wp = nxt
            remaining -= step_m

    def _route_target_waypoint(self):
        if self._ego_lane_route_wp is None:
            return None
        return _waypoint_lookahead_from(self._ego_lane_route_wp, _ROUTE_LOOKAHEAD_M)

    def _drive_along_merge_route(
        self, speed_mps: float, throttle: float = 0.35, *, lane_change: bool = True
    ):
        """Drive along ego-lane route with a shrinking left-side lateral offset."""
        target_wp = self._route_target_waypoint()
        if target_wp is None:
            return

        right = target_wp.transform.get_right_vector()
        target_loc = target_wp.transform.location + right * (-self.lateral_offset_m)
        target_tf = carla.Transform(target_loc, target_wp.transform.rotation)

        vehicle = self.pullout_vehicle
        veh_tf = vehicle.get_transform()
        forward = veh_tf.get_forward_vector()
        veh_right = veh_tf.get_right_vector()
        lat_err = lc.lateral_error_m(
            veh_tf.location.x,
            veh_tf.location.y,
            target_tf.location.x,
            target_tf.location.y,
            veh_right.x,
            veh_right.y,
        )
        yaw_err = _normalize_yaw_deg(target_tf.rotation.yaw - veh_tf.rotation.yaw)
        max_steer = lc.speed_scaled_max_steer(speed_mps, lane_change=lane_change)
        if lane_change:
            steer = lc.compute_centering_steer(
                lat_err,
                yaw_err,
                lat_gain=lc.CENTER_LAT_GAIN,
                yaw_gain=lc.CENTER_YAW_GAIN,
                max_steer=max_steer,
            )
        else:
            steer = lc.compute_steer(
                lat_err,
                yaw_err,
                lat_gain=lc.LAT_GAIN,
                yaw_gain=lc.YAW_GAIN,
                max_steer=max_steer,
                lateral_weight=lc.lateral_weight_for_yaw(yaw_err),
            )

        vehicle.set_target_velocity(
            carla.Vector3D(
                x=forward.x * speed_mps,
                y=forward.y * speed_mps,
                z=0.0,
            )
        )
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle, steer=steer, brake=0.0, hand_brake=False
            )
        )

    def _drive_route_npc(self, route_wp, speed_mps: float, throttle: float = 0.28):
        route_wp = self._reanchor_route_wp(route_wp)
        veh_loc = self.pullout_vehicle.get_transform().location
        if veh_loc.distance(route_wp.transform.location) < _LANE_ROUTE_ADVANCE_M:
            route_wp = self._advance_route_wp(route_wp, _STEP_M)
        lookahead = _waypoint_lookahead_from(route_wp, _ROUTE_LOOKAHEAD_M)
        _drive_along_waypoint(self.pullout_vehicle, lookahead, speed_mps, throttle=throttle)
        return route_wp

    def _drive_source_lane_npc(self, speed_mps: float):
        self._source_route_wp = self._drive_route_npc(
            self._source_route_wp,
            speed_mps,
            throttle=0.28,
        )

    def _spawn_on_roadside(self, blueprint, ahead_wp):
        rotation = ahead_wp.transform.rotation
        base = ahead_wp.transform.location

        left_lane_wp = ahead_wp.get_left_lane()
        if left_lane_wp is not None and left_lane_wp.lane_type == carla.LaneType.Driving:
            actor = self.world.try_spawn_actor(
                blueprint,
                carla.Transform(
                    left_lane_wp.transform.location + carla.Location(z=0.6),
                    left_lane_wp.transform.rotation,
                ),
            )
            if actor is not None:
                self._source_route_wp = left_lane_wp
                logger.info("Scenario 06: red pullout vehicle spawned in left adjacent lane.")
                return actor

        right = ahead_wp.transform.get_right_vector()
        for offset_m in (_ROADSIDE_OFFSET_M, 2.7, 2.4, 3.3, 2.0, 1.5):
            location = base + right * (-offset_m)
            actor = self.world.try_spawn_actor(
                blueprint,
                carla.Transform(location + carla.Location(z=0.6), rotation),
            )
            if actor is not None:
                logger.info(
                    "Scenario 06: pullout vehicle spawned %.1fm left of lane center.",
                    offset_m,
                )
                return actor

        spawn_tf = carla.Transform(
            ahead_wp.transform.location + carla.Location(z=0.5),
            ahead_wp.transform.rotation,
        )
        actor = self.world.try_spawn_actor(blueprint, spawn_tf)
        if actor is not None:
            logger.warning(
                "Scenario 06: pullout vehicle spawned in-lane as fallback at %.1fm ahead.",
                self.npc_start_distance_m,
            )
        return actor

    def teardown(self):
        if self.pullout_vehicle and self.pullout_vehicle.is_alive:
            try:
                self.pullout_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            except Exception as exc:
                logger.warning("Scenario 06: could not stop pullout vehicle: %s", exc)
        super().teardown()


