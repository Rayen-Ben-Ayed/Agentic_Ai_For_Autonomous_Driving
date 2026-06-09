import os

import carla
import logging

from pipeline_log import log_stage
from simulation import lane_controller as lc
from simulation.timing_config import STEP_INTERVAL_S

logger = logging.getLogger(__name__)

DEFAULT_MAX_SPEED_MPS = float(os.getenv("ACTION_MAX_SPEED_MPS", "8.0"))
YIELD_SPEED_FACTOR = float(os.getenv("YIELD_SPEED_FACTOR", "0.5"))
MIN_FOLLOW_FROM_REST_MPS = float(os.getenv("MIN_FOLLOW_FROM_REST_MPS", "3.5"))
FOLLOW_CRUISE_MPS = float(os.getenv("FOLLOW_CRUISE_MPS", str(MIN_FOLLOW_FROM_REST_MPS)))
FOLLOW_SUSTAIN_THROTTLE = float(os.getenv("FOLLOW_SUSTAIN_THROTTLE", "0.32"))
STATIONARY_SPEED_MPS = float(os.getenv("STATIONARY_SPEED_MPS", "0.5"))
SPEED_KP = float(os.getenv("ACTION_SPEED_KP", "0.25"))
MAX_THROTTLE = float(os.getenv("ACTION_MAX_THROTTLE", "0.6"))
MAX_BRAKE = float(os.getenv("ACTION_MAX_BRAKE", "0.85"))
YIELD_MIN_BRAKE = float(os.getenv("YIELD_MIN_BRAKE", "0.45"))
# Consecutive agent steps with the same lateral action before giving up
# (each step commits controls for STEP_INTERVAL_S simulated seconds).
_default_lane_change_steps = max(1, round(12.0 / STEP_INTERVAL_S))
LANE_CHANGE_MAX_STEPS = int(
    os.getenv("LANE_CHANGE_MAX_STEPS", str(_default_lane_change_steps))
)
OVERTAKE_SPEED_FACTOR = float(os.getenv("OVERTAKE_SPEED_FACTOR", "1.1"))


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
        self._current_maneuver: str | None = None
        self._maneuver_steps: int = 0

    def _clear_maneuver(self) -> None:
        self._centering_side = None
        self._current_maneuver = None
        self._maneuver_steps = 0

    def _clear_lateral_tracking(self) -> None:
        """Stop an in-progress lane change without affecting lane-centering."""
        self._current_maneuver = None
        self._maneuver_steps = 0

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

    def _follow_lane_target(self, current_speed: float) -> float:
        """Accelerate up to cruise speed, then hold cruise (capped at max)."""
        cruise = FOLLOW_CRUISE_MPS
        cap = DEFAULT_MAX_SPEED_MPS
        if current_speed < cruise:
            return min(cap, cruise)
        return min(cap, current_speed)

    def _yield_target(self, current_speed: float) -> float:
        """Halve speed when moving; hold stopped when already near rest."""
        if current_speed < STATIONARY_SPEED_MPS:
            return 0.0
        halved = current_speed * YIELD_SPEED_FACTOR
        # Always command strictly slower than now (never throttle on yield).
        return max(0.0, min(halved, current_speed - 0.15))

    def _maneuver_target_speed(
        self,
        current_speed: float,
        *,
        overtake: bool = False,
        lateral_error_m: float = 0.0,
    ) -> float:
        """Lane changes avoid accelerating into hazards; overtake allows a modest bump."""
        if overtake and current_speed >= STATIONARY_SPEED_MPS:
            boosted = min(
                DEFAULT_MAX_SPEED_MPS,
                current_speed * OVERTAKE_SPEED_FACTOR,
            )
            return max(current_speed, boosted)

        if current_speed < STATIONARY_SPEED_MPS:
            return MIN_FOLLOW_FROM_REST_MPS * 0.55

        # Large lateral offset: creep, do not add longitudinal speed.
        if abs(lateral_error_m) > 1.5:
            return min(current_speed, MIN_FOLLOW_FROM_REST_MPS * 0.7)

        return min(DEFAULT_MAX_SPEED_MPS, current_speed)

    def _drive_with_target_speed(
        self,
        ego_vehicle,
        target_speed: float,
        steer: float,
        *,
        brake_only: bool = False,
        sustain_cruise: bool = False,
    ) -> carla.VehicleControl:
        control = carla.VehicleControl()
        current_speed = self._ego_speed(ego_vehicle)
        speed_error = target_speed - current_speed

        if brake_only:
            control.throttle = 0.0
            if current_speed < STATIONARY_SPEED_MPS:
                control.brake = min(MAX_BRAKE, YIELD_MIN_BRAKE)
            else:
                control.brake = min(
                    MAX_BRAKE,
                    max(YIELD_MIN_BRAKE, SPEED_KP * abs(speed_error)),
                )
        elif speed_error > 0.08:
            control.throttle = min(MAX_THROTTLE, SPEED_KP * speed_error)
            control.brake = 0.0
        elif speed_error < -0.08:
            control.throttle = 0.0
            control.brake = min(MAX_BRAKE, SPEED_KP * abs(speed_error))
        elif (
            sustain_cruise
            and current_speed >= STATIONARY_SPEED_MPS
            and current_speed < DEFAULT_MAX_SPEED_MPS
        ):
            # At cruise the P-term is near zero; CARLA still needs throttle to hold speed.
            control.throttle = min(MAX_THROTTLE, FOLLOW_SUSTAIN_THROTTLE)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = 0.0

        control.steer = steer
        control.hand_brake = False
        control.reverse = False
        return control

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

    def _begin_lane_maneuver(self, action: str) -> None:
        if action == "change_lane_left":
            self._centering_side = "left"
            self._current_maneuver = "change_lane_left"
        elif action == "change_lane_right":
            self._centering_side = "right"
            self._current_maneuver = "change_lane_right"
        elif action == "overtake":
            self._centering_side = "left"
            self._current_maneuver = "overtake"
        self._maneuver_steps += 1

    def _target_waypoint(self, ego_vehicle, action: str):
        base_wp = self._driving_waypoint(ego_vehicle)
        if action == "change_lane_left":
            return self._adjacent_lane_waypoint(base_wp, "left")
        if action == "change_lane_right":
            return self._adjacent_lane_waypoint(base_wp, "right")
        if action == "overtake":
            return self._adjacent_lane_waypoint(base_wp, "left")

        if self._centering_side and action in ("follow_lane", "yield", "stop"):
            adj = self._adjacent_lane_waypoint(base_wp, self._centering_side)
            if adj:
                return adj
            self._centering_side = None

        return self._lookahead(base_wp)

    def _in_target_lane(self, ego_vehicle, heading_wp) -> bool:
        """True when the ego map waypoint matches the maneuver target lane."""
        if heading_wp is None or self._centering_side is None:
            return False
        ego_wp = self._driving_waypoint(ego_vehicle)
        if ego_wp is None:
            return False
        return (
            ego_wp.road_id == heading_wp.road_id
            and ego_wp.lane_id == heading_wp.lane_id
        )

    def _lane_steer(self, ego_vehicle, action: str) -> tuple[float, float]:
        base_wp = self._driving_waypoint(ego_vehicle)
        heading_wp = self._target_waypoint(ego_vehicle, action)
        lane_wp = base_wp
        if action in ("change_lane_left", "change_lane_right", "overtake") or (
            self._centering_side and action in ("follow_lane", "yield", "stop")
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
        centered = abs(lat_err) <= lc.CENTER_TOLERANCE_M
        in_target_lane = self._in_target_lane(ego_vehicle, heading_wp)
        if (
            self._centering_side
            and action in ("follow_lane", "yield", "stop")
            and centered
            and (in_target_lane or heading_wp is None)
        ):
            self._clear_maneuver()
        return steer, lat_err

    def _lane_change_fallback(self, ego_vehicle, action: str) -> carla.VehicleControl | None:
        """If adjacent lane is missing, follow current lane instead of steering off-road."""
        if action not in ("change_lane_left", "change_lane_right", "overtake"):
            return None
        side = "left" if action in ("change_lane_left", "overtake") else "right"
        base_wp = self._driving_waypoint(ego_vehicle)
        if self._adjacent_lane_waypoint(base_wp, side) is not None:
            return None

        logger.warning("No %s driving lane; falling back to follow_lane.", side)
        self._clear_maneuver()
        steer, _ = self._lane_steer(ego_vehicle, "follow_lane")
        speed = self._ego_speed(ego_vehicle)
        target = self._follow_lane_target(speed)
        return self._drive_with_target_speed(
            ego_vehicle, target, steer, sustain_cruise=True
        )

    def execute_action(self, action: str):
        if action not in self.valid_actions:
            logger.error("Invalid action: %s", action)
            return False

        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not initialized.")
            return False

        current_speed = self._ego_speed(ego_vehicle)
        target_speed = current_speed

        if action == "stop":
            self._clear_lateral_tracking()
            steer, lat_err = self._lane_steer(ego_vehicle, action)
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = steer
            control.brake = 1.0
            if lat_err != 0.0:
                log_stage(
                    logger,
                    "CARLA",
                    "lane_hold action=stop lat_err=%.2fm steer=%.3f",
                    lat_err,
                    steer,
                )
        elif action == "yield":
            self._clear_lateral_tracking()
            steer, lat_err = self._lane_steer(ego_vehicle, action)
            target_speed = self._yield_target(current_speed)
            control = self._drive_with_target_speed(
                ego_vehicle,
                target_speed,
                steer,
                brake_only=True,
            )
        elif action == "follow_lane":
            self._clear_lateral_tracking()
            steer, _ = self._lane_steer(ego_vehicle, action)
            target_speed = self._follow_lane_target(current_speed)
            control = self._drive_with_target_speed(
                ego_vehicle,
                target_speed,
                steer,
                sustain_cruise=True,
            )
        elif action in ("change_lane_left", "change_lane_right", "overtake"):
            fallback = self._lane_change_fallback(ego_vehicle, action)
            if fallback is not None:
                control = fallback
                target_speed = self._follow_lane_target(current_speed)
            else:
                if (
                    self._current_maneuver == action
                    and self._maneuver_steps >= LANE_CHANGE_MAX_STEPS
                ):
                    logger.warning(
                        "Maneuver %s exceeded %d steps; clearing.",
                        action,
                        LANE_CHANGE_MAX_STEPS,
                    )
                    self._clear_maneuver()

                self._begin_lane_maneuver(action)
                steer, lat_err = self._lane_steer(ego_vehicle, action)
                target_speed = self._maneuver_target_speed(
                    current_speed,
                    overtake=(action == "overtake"),
                    lateral_error_m=lat_err,
                )
                control = self._drive_with_target_speed(
                    ego_vehicle, target_speed, steer
                )
                if lat_err != 0.0:
                    log_stage(
                        logger,
                        "CARLA",
                        "lane_target side=%s lat_err=%.2fm steer=%.3f",
                        self._centering_side,
                        lat_err,
                        steer,
                    )
        else:
            logger.error("Unhandled action: %s", action)
            return False

        ego_vehicle.apply_control(control)
        log_stage(
            logger,
            "CARLA",
            "apply_control action=%s speed=%.2f target=%.2f throttle=%.2f steer=%.2f brake=%.2f",
            action,
            current_speed,
            target_speed,
            control.throttle,
            control.steer,
            control.brake,
        )
        return True
