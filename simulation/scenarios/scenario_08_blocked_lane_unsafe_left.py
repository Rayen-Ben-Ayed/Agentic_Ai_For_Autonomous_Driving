import logging

import carla

from .scenario_07_blocked_lane_clear_left import (
    Scenario07BlockedLaneClearLeft,
    _advance_along_lane,
    _find_blueprint,
    _waypoint_ahead_on_lane,
)

logger = logging.getLogger(__name__)


class Scenario08BlockedLaneUnsafeLeft(Scenario07BlockedLaneClearLeft):
    def __init__(self, carla_client):
        super().__init__(carla_client)
        self.left_lane_vehicle = None
        self.left_vehicle_speed = 5.5
        self.left_vehicle_distance_m = 42.0
        self.ego_throttle = 0.52

    def setup(self):
        """Stopped vehicle ahead, plus moving traffic in the left lane."""
        super().setup()
        ego = self.carla_client.get_ego_vehicle()
        if not ego:
            return

        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            ego.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            return

        left_wp = ego_wp.get_left_lane()
        if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
            logger.warning("Scenario 08: no left lane available for unsafe-lane vehicle.")
            return

        spawn_wp, _ = _waypoint_ahead_on_lane(left_wp, self.left_vehicle_distance_m)
        if spawn_wp is None:
            wp = left_wp
            for _ in range(10):
                nxt = _advance_along_lane(wp, 2.0)
                if nxt is None:
                    break
                wp = nxt
            spawn_wp = wp

        spawn_transform = spawn_wp.transform
        spawn_transform.location.z += 0.5
        blueprint_library = self.world.get_blueprint_library()
        bp = _find_blueprint(
            blueprint_library,
            [
                "vehicle.bmw.grandtourer",
                "vehicle.nissan.patrol_2021",
                "vehicle.chevrolet.impala",
                "vehicle.tesla.model3",
            ],
        )
        self.left_lane_vehicle = self.world.try_spawn_actor(bp, spawn_transform)
        if self.left_lane_vehicle is None:
            logger.error("Scenario 08: failed to spawn the left-lane vehicle.")
            return

        self.npc_actors.append(self.left_lane_vehicle)
        self.left_lane_vehicle.set_autopilot(False)
        self._drive_left_lane_vehicle()

        self.world.debug.draw_string(
            self.left_lane_vehicle.get_location() + carla.Location(z=3.0),
            "LEFT LANE OCCUPIED",
            color=carla.Color(255, 180, 0),
            life_time=30.0,
        )
        logger.info("=================================================")
        logger.info("SCENARIO 08: BLOCKED LANE, LEFT LANE UNSAFE")
        logger.info("No-agent: ego drives into the stopped vehicle.")
        logger.info("Agent: expected response is yield/stop, not change_lane_left.")
        logger.info("=================================================")

    def update(self, step=None, *, allow_trigger=True):
        super().update(step, allow_trigger=allow_trigger)
        if self.left_lane_vehicle and self.left_lane_vehicle.is_alive:
            self._drive_left_lane_vehicle()

    def _drive_left_lane_vehicle(self):
        transform = self.left_lane_vehicle.get_transform()
        forward = transform.get_forward_vector()
        self.left_lane_vehicle.set_target_velocity(
            carla.Vector3D(
                x=forward.x * self.left_vehicle_speed,
                y=forward.y * self.left_vehicle_speed,
                z=0.0,
            )
        )
        self.left_lane_vehicle.apply_control(
            carla.VehicleControl(throttle=0.45, steer=0.0, brake=0.0)
        )
