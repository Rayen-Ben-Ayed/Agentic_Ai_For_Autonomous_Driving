import logging

import carla

from .base_scenario import BaseScenario
from .scenario_07_blocked_lane_clear_left import (
    _ego_frame_offsets,
    _find_blueprint,
    _lane_is_clear_midroad,
    _waypoint_ahead_on_lane,
)

logger = logging.getLogger(__name__)

_NO_JUNCTION_M = 85.0


def _same_direction_lane(reference_wp, candidate_wp):
    return (
        candidate_wp is not None
        and candidate_wp.lane_type == carla.LaneType.Driving
        and reference_wp.lane_id * candidate_wp.lane_id > 0
    )


def select_right_lane_pullout_spawn_point(world):
    """Select an urban mid-road spawn suitable for a roadside parked car."""
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

        if not _lane_is_clear_midroad(ego_wp, _NO_JUNCTION_M):
            continue

        hazard_wp, _ = _waypoint_ahead_on_lane(ego_wp, 62.0)
        if hazard_wp is None or hazard_wp.is_junction:
            continue

        candidates.append((index, spawn_point))

    if not candidates:
        logger.warning(
            "Scenario 06: no ideal right-lane pullout spawn found; using CARLA default."
        )
        return None

    original_index, spawn_point = candidates[min(len(candidates) - 1, len(candidates) // 3)]
    logger.info(
        "Scenario 06 selected mid-road ego spawn #%d from %s.",
        original_index,
        world.get_map().name,
    )
    return spawn_point


class Scenario06RightLanePullout(BaseScenario):
    def __init__(self, carla_client):
        super().__init__(carla_client)
        self.pullout_vehicle = None
        self.control_ego = False
        self.ego_throttle = 0.53
        self.vehicle_start_distance_m = 58.0
        self.merge_target_distance_m = 76.0
        self.trigger_distance_m = 40.0
        self.parking_side_offset_m = 3.2
        self.pullout_speed_mps = 6.2
        self.merge_started = False
        self.merge_target_location = None
        self.ego_lane_forward = None
        self._last_logged_bucket = None

    def setup(self):
        """Vehicle waits at the roadside/parking side, then pulls into ego lane."""
        ego = self.carla_client.get_ego_vehicle()
        if not ego:
            logger.error("Ego vehicle not found. Cannot setup scenario 06.")
            return

        ego_tf = ego.get_transform()
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            ego_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            logger.error("Scenario 06: no ego lane waypoint.")
            return

        ego_hazard_wp, _ = _waypoint_ahead_on_lane(
            ego_wp, self.merge_target_distance_m
        )
        roadside_wp, actual_m = _waypoint_ahead_on_lane(
            ego_wp, self.vehicle_start_distance_m
        )
        if ego_hazard_wp is None or roadside_wp is None:
            logger.error("Scenario 06: failed to find pullout lane waypoints.")
            return

        longitudinal, lateral = _ego_frame_offsets(
            ego_tf, ego_hazard_wp.transform.location
        )
        if longitudinal <= 0.0 or abs(lateral) > 3.0:
            logger.error("Scenario 06: selected hazard point is not ahead of ego.")
            return

        self.merge_target_location = ego_hazard_wp.transform.location
        self.merge_target_location.z += 0.5
        self.ego_lane_forward = ego_hazard_wp.transform.get_forward_vector()

        blueprint_library = self.world.get_blueprint_library()
        bp = _find_blueprint(
            blueprint_library,
            [
                "vehicle.mini.cooper_s_2021",
                "vehicle.volkswagen.t2_2021",
                "vehicle.carlamotors.carlacola",
                "vehicle.nissan.micra",
            ],
        )

        self.pullout_vehicle = self._spawn_parked_vehicle(bp, roadside_wp)
        if self.pullout_vehicle is None:
            logger.error("Scenario 06: failed to spawn roadside parked vehicle.")
            return

        self.npc_actors.append(self.pullout_vehicle)
        self.pullout_vehicle.set_autopilot(False)
        self._hold_pullout_vehicle()

        self.world.debug.draw_string(
            self.pullout_vehicle.get_location() + carla.Location(z=3.0),
            "PARKED CAR WILL PULL OUT",
            color=carla.Color(255, 180, 0),
            life_time=30.0,
        )
        self.world.debug.draw_string(
            self.merge_target_location + carla.Location(z=3.0),
            "MERGE CONFLICT POINT",
            color=carla.Color(255, 0, 0),
            life_time=30.0,
        )

        logger.info("=================================================")
        logger.info("SCENARIO 06: PARKED CAR PULLS INTO EGO LANE")
        logger.info("Second car is parked roadside %.1fm ahead.", actual_m)
        logger.info("It merges when ego is within %.1fm of the conflict point.", self.trigger_distance_m)
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
        npc_distance = self.pullout_vehicle.get_location().distance(
            self.merge_target_location
        )

        if not self.merge_started:
            self._hold_pullout_vehicle()
            if allow_trigger and ego_distance < self.trigger_distance_m:
                self.merge_started = True
                logger.info(
                    "Scenario 06: pullout starts now (ego->conflict=%.1fm).",
                    ego_distance,
                )
                self._drive_pullout_vehicle()
        else:
            self._drive_pullout_vehicle()

        bucket = int(ego_distance // 5)
        if bucket != self._last_logged_bucket:
            self._last_logged_bucket = bucket
            logger.info(
                "Scenario 06 | ego->conflict=%.1f m | npc->conflict=%.1f m | pullout=%s",
                ego_distance,
                npc_distance,
                self.merge_started,
            )

    def is_llm_needed(self, world_state):
        if self.llm_queried:
            return False
        for actor in world_state.get("nearby_actors", []):
            if actor.get("is_scenario_npc") and actor.get("distance", 999.0) < 45.0:
                self.llm_queried = True
                logger.info(
                    "Scenario 06: pullout vehicle detected at %.1fm, LLM should decide.",
                    actor.get("distance"),
                )
                return True
        return False

    def _hold_pullout_vehicle(self):
        self.pullout_vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )

    def _spawn_parked_vehicle(self, blueprint, roadside_wp):
        right = roadside_wp.transform.get_right_vector()
        for side_sign in (1.0, -1.0):
            actor = self._try_spawn_parked_vehicle_side(
                blueprint,
                roadside_wp,
                right,
                side_sign,
            )
            if actor is not None:
                return actor
        return None

    def _try_spawn_parked_vehicle_side(self, blueprint, roadside_wp, right, side_sign):
        for offset_m in (
            self.parking_side_offset_m,
            self.parking_side_offset_m - 0.6,
            self.parking_side_offset_m - 1.1,
            self.parking_side_offset_m - 1.6,
            1.2,
        ):
            signed_offset = offset_m * side_sign
            location = roadside_wp.transform.location + right * signed_offset
            location.z += 0.6
            spawn_transform = carla.Transform(location, roadside_wp.transform.rotation)
            actor = self.world.try_spawn_actor(blueprint, spawn_transform)
            if actor is not None:
                logger.info(
                    "Scenario 06: parked car spawned %.1fm from ego lane.",
                    signed_offset,
                )
                return actor
        return None

    def _drive_pullout_vehicle(self):
        location = self.pullout_vehicle.get_location()
        to_target = carla.Vector3D(
            self.merge_target_location.x - location.x,
            self.merge_target_location.y - location.y,
            0.0,
        )
        distance = (to_target.x * to_target.x + to_target.y * to_target.y) ** 0.5
        if distance > 1.5:
            direction = carla.Vector3D(to_target.x / distance, to_target.y / distance, 0.0)
        else:
            direction = self.ego_lane_forward

        self.pullout_vehicle.set_target_velocity(
            carla.Vector3D(
                x=direction.x * self.pullout_speed_mps,
                y=direction.y * self.pullout_speed_mps,
                z=0.0,
            )
        )
        self.pullout_vehicle.apply_control(
            carla.VehicleControl(throttle=0.65, steer=0.0, brake=0.0, hand_brake=False)
        )

    def teardown(self):
        if self.pullout_vehicle and self.pullout_vehicle.is_alive:
            try:
                self.pullout_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            except Exception as e:
                logger.warning("Scenario 06: could not stop pullout vehicle: %s", e)
        super().teardown()
