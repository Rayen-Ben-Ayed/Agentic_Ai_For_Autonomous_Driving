import carla
import logging

from pipeline_log import log_stage
from simulation import lane_controller as lc

logger = logging.getLogger(__name__)


class ActionExecutor:
    def __init__(self, carla_client):
        self.carla_client = carla_client
        self.valid_actions = [
            "overtake",
            "follow_lane",
            "stop",
            "yield",
            "change_lane_left",
            "change_lane_right",
        ]
        # After a lateral maneuver, keep steering toward that lane until centered.
        self._centering_side: str | None = None

    def _driving_waypoint(self, ego_vehicle):
        world = self.carla_client.get_world()
        if not world:
            return None
        transform = ego_vehicle.get_transform()
        return world.get_map().get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def _lookahead(self, waypoint, distance_m: float | None = None):
        if not waypoint:
            return None
        dist = distance_m if distance_m is not None else lc.LOOKAHEAD_M
        ahead = waypoint.next(dist)
        return ahead[0] if ahead else waypoint

    def _adjacent_lane_waypoint(self, waypoint, side: str):
        if not waypoint:
            return None
        if side == "left":
            lane_wp = waypoint.get_left_lane()
        else:
            lane_wp = waypoint.get_right_lane()
        if lane_wp is None or lane_wp.lane_type != carla.LaneType.Driving:
            return None
        return self._lookahead(lane_wp)

    def _ego_speed(self, ego_vehicle) -> float:
        v = ego_vehicle.get_velocity()
        return (v.x**2 + v.y**2 + v.z**2) ** 0.5

    def _steer_toward_waypoint(
        self,
        ego_vehicle,
        lane_wp,
        heading_wp,
        *,
        lane_change: bool,
    ) -> tuple[float, float]:
        """Lateral error vs current lane; yaw error vs lookahead heading."""
        if not lane_wp:
            return 0.0, 0.0

        ego_tf = ego_vehicle.get_transform()
        lane_tf = lane_wp.transform
        heading_tf = (heading_wp or lane_wp).transform
        right = lane_tf.get_right_vector()
        lat_err = lc.lateral_error_m(
            ego_tf.location.x,
            ego_tf.location.y,
            lane_tf.location.x,
            lane_tf.location.y,
            right.x,
            right.y,
        )
        yaw_err = lc.normalize_yaw_error(
            heading_tf.rotation.yaw - ego_tf.rotation.yaw
        )
        speed = self._ego_speed(ego_vehicle)
        max_steer = lc.speed_scaled_max_steer(speed, lane_change=lane_change)
        steer = lc.compute_steer(
            lat_err,
            yaw_err,
            lat_gain=lc.LAT_GAIN,
            yaw_gain=lc.YAW_GAIN,
            max_steer=max_steer,
            lateral_weight=lc.lateral_weight_for_yaw(yaw_err),
        )
        return steer, lat_err

    def _target_waypoint(self, ego_vehicle, action: str):
        base_wp = self._driving_waypoint(ego_vehicle)
        if action == "change_lane_left":
            self._centering_side = "left"
            return self._adjacent_lane_waypoint(base_wp, "left")
        if action == "change_lane_right":
            self._centering_side = "right"
            return self._adjacent_lane_waypoint(base_wp, "right")
        if action == "overtake":
            self._centering_side = "left"
            return self._adjacent_lane_waypoint(base_wp, "left")

        if self._centering_side and action in ("follow_lane", "yield"):
            adj = self._adjacent_lane_waypoint(base_wp, self._centering_side)
            if adj:
                return adj
            self._centering_side = None

        return self._lookahead(base_wp)

    def _lane_steer(self, ego_vehicle, action: str) -> tuple[float, float]:
        base_wp = self._driving_waypoint(ego_vehicle)
        heading_wp = self._target_waypoint(ego_vehicle, action)
        lane_wp = base_wp
        if action in ("change_lane_left", "change_lane_right", "overtake") or (
            self._centering_side and action in ("follow_lane", "yield")
        ):
            lane_wp = heading_wp or base_wp
        lane_change = action in (
            "change_lane_left",
            "change_lane_right",
            "overtake",
        ) or self._centering_side is not None
        steer, lat_err = self._steer_toward_waypoint(
            ego_vehicle,
            lane_wp,
            heading_wp or base_wp,
            lane_change=lane_change,
        )
        if (
            self._centering_side
            and action in ("follow_lane", "yield")
            and abs(lat_err) <= lc.CENTER_TOLERANCE_M
        ):
            self._centering_side = None
        return steer, lat_err

    def _throttle_for_action(self, action: str, lateral_error_m: float) -> float:
        if action == "stop":
            return 0.0
        if action == "yield":
            return 0.0
        base = {
            "follow_lane": 0.35,
            "change_lane_left": 0.26,
            "change_lane_right": 0.26,
            "overtake": 0.35,
        }.get(action, 0.3)
        if action in ("change_lane_left", "change_lane_right", "overtake"):
            scale = max(0.35, 1.0 - abs(lateral_error_m) / 5.0)
            return base * scale
        if action == "follow_lane" and abs(lateral_error_m) > 1.5:
            return base * 0.75
        return base

    def execute_action(self, action: str):
        if action not in self.valid_actions:
            logger.error("Invalid action: %s", action)
            return False

        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not initialized.")
            return False

        if action == "stop":
            self._centering_side = None

        steer, lat_err = self._lane_steer(ego_vehicle, action)
        control = carla.VehicleControl()

        if action == "follow_lane":
            control.throttle = self._throttle_for_action(action, lat_err)
            control.steer = steer
            control.brake = 0.0
        elif action == "stop":
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 1.0
        elif action == "yield":
            control.throttle = 0.0
            control.steer = steer
            control.brake = 0.6
        elif action in ("change_lane_left", "change_lane_right", "overtake"):
            control.throttle = self._throttle_for_action(action, lat_err)
            control.steer = steer
            control.brake = 0.0
            if lat_err != 0.0:
                log_stage(
                    logger,
                    "CARLA",
                    "lane_target side=%s lat_err=%.2fm steer=%.3f",
                    self._centering_side,
                    lat_err,
                    steer,
                )

        ego_vehicle.apply_control(control)
        log_stage(
            logger,
            "CARLA",
            "apply_control action=%s throttle=%.2f steer=%.2f brake=%.2f",
            action,
            control.throttle,
            control.steer,
            control.brake,
        )
        return True
