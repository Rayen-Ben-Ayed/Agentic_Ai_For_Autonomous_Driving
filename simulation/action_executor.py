import os

import carla
import logging

from pipeline_log import log_stage
from simulation import junction_planner as jp
from simulation import lane_change_controller as lcc
from simulation import lane_controller as lc
from simulation import maneuver_planner as mp
from simulation import telemetry
from simulation.timing_config import STEP_INTERVAL_S

logger = logging.getLogger(__name__)

DEFAULT_MAX_SPEED_MPS = float(os.getenv("ACTION_MAX_SPEED_MPS", "8.0"))
YIELD_SPEED_FACTOR = float(os.getenv("YIELD_SPEED_FACTOR", "0.5"))
MIN_FOLLOW_FROM_REST_MPS = float(os.getenv("MIN_FOLLOW_FROM_REST_MPS", "3.5"))
FOLLOW_CRUISE_MPS = float(os.getenv("FOLLOW_CRUISE_MPS", str(MIN_FOLLOW_FROM_REST_MPS)))
FOLLOW_SUSTAIN_THROTTLE = float(os.getenv("FOLLOW_SUSTAIN_THROTTLE", "0.32"))
# Minimum throttle while accelerating toward target speed. The bare P-term decays
# near cruise (e.g. 0.25*0.7=0.17) and the ego plateaus below target; this floor
# ensures it actually reaches cruise.
FOLLOW_ACCEL_MIN_THROTTLE = float(os.getenv("FOLLOW_ACCEL_MIN_THROTTLE", "0.35"))
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
# Step used to walk a frozen lane anchor forward without letting waypoint.next()
# pick an arbitrary junction connector (which corrupts the merge reference).
LANE_ADVANCE_STEP_M = float(os.getenv("LANE_ADVANCE_STEP_M", "2.0"))


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
            *jp.JUNCTION_ACTIONS,
        ]
        # After a lateral maneuver, keep steering toward that lane until centered.
        self._centering_side: str | None = None
        self._target_lane_id: int | None = None
        self._current_maneuver: str | None = None
        self._maneuver_steps: int = 0
        # Per-step active maneuver, recomputed every physics tick by tick().
        self._active_action: str | None = None
        self._plan: mp.ManeuverPlan | None = None
        self._centering_plan: mp.ManeuverPlan | None = None
        # Frozen junction path (approach + connector + exit) and progress index.
        # Committed once by go_straight/turn_right/turn_left, then persists across
        # follow_lane/yield/stop steps (they steer along it) until it completes or
        # a lateral lane change discards it.
        self._junction_plan: jp.JunctionPlan | None = None
        self._junction_idx: int = 0
        # Set in begin_action when a junction action was requested but no connector
        # exists for that direction (distinguishes "hold a safe stop" from "plan
        # completed mid-step, drive on normally").
        self._junction_action_infeasible: bool = False
        self._elapsed_s: float = 0.0
        self._merge_logged: bool = False
        # Telemetry / transparency bookkeeping.
        self._last_steer_info: dict = {"yaw_err": 0.0, "max_steer": 0.0}
        self._step_index: int = 0
        self._total_ticks: int = 0
        self._tick_index: int = 0
        self._last_telem_lane_id: int | None = None

    def set_step_context(self, step_index: int, total_ticks: int) -> None:
        """Called by the main loop each step so telemetry rows carry step/tick context."""
        self._step_index = step_index
        self._total_ticks = total_ticks
        self._tick_index = 0
        self._last_telem_lane_id = None

    def _clear_maneuver(self) -> None:
        self._centering_side = None
        self._target_lane_id = None
        self._centering_plan = None
        self._current_maneuver = None
        self._maneuver_steps = 0

    def _active_centering_plan(self) -> mp.ManeuverPlan | None:
        return self._centering_plan or self._plan

    def is_lane_centering_active(self) -> bool:
        return self._centering_side is not None

    def lane_centering_snapshot(self) -> dict:
        return {
            "lane_centering_incomplete": self.is_lane_centering_active(),
            "lane_centering_side": self._centering_side,
        }

    def is_junction_committed(self) -> bool:
        return self._junction_plan is not None

    def junction_snapshot(self) -> dict:
        """Round-1 junction decision state, so the agent knows to move to round 2
        (drive with follow_lane/yield/stop/change_lane_*) instead of re-deciding."""
        return {
            "junction_committed": self.is_junction_committed(),
            "junction_committed_direction": (
                self._junction_plan.direction if self._junction_plan else None
            ),
        }

    def frozen_centering_lat_err_m(self, ego_vehicle) -> float | None:
        errors = self._frozen_centering_errors(ego_vehicle)
        return errors[0] if errors else None

    def _frozen_centering_errors(self, ego_vehicle) -> tuple[float, float] | None:
        """Lateral/yaw error vs frozen target-lane anchor (junction-safe)."""
        plan = self._active_centering_plan()
        if not plan or plan.target_anchor_x is None:
            return None
        travel = self._merge_travel_m(ego_vehicle, plan)
        tgt_la = self._frozen_lane_lookahead(plan, "target", travel)
        tgt_pose = lcc.lane_pose_from_waypoint(tgt_la)
        if not tgt_pose:
            return None
        ego_tf = ego_vehicle.get_transform()
        return lcc.errors_in_target_frame(
            ego_tf.location.x,
            ego_tf.location.y,
            ego_tf.rotation.yaw,
            tgt_pose,
        )

    def _merge_travel_m(self, ego_vehicle, plan: mp.ManeuverPlan) -> float:
        if plan.start_ego_x is None or plan.source_anchor_yaw is None:
            return 0.0
        tf = ego_vehicle.get_transform()
        return mp.forward_travel_m(
            tf.location.x,
            tf.location.y,
            plan.start_ego_x,
            plan.start_ego_y,
            plan.source_anchor_yaw,
        )

    def _lat_yaw_vs_waypoint(self, ego_vehicle, lane_wp, heading_wp=None):
        if not lane_wp:
            return 0.0, 0.0
        heading_wp = heading_wp or lane_wp
        ego_tf = ego_vehicle.get_transform()
        lane_tf = lane_wp.transform
        heading_tf = heading_wp.transform
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
        return lat_err, yaw_err

    def _is_laterally_centered(self, ego_vehicle, heading_wp) -> bool:
        if heading_wp is None:
            return False
        lat_err, yaw_err = self._lat_yaw_vs_waypoint(
            ego_vehicle, heading_wp, heading_wp
        )
        return (
            abs(lat_err) <= lc.CENTER_TOLERANCE_M
            and abs(yaw_err) <= lc.CENTER_YAW_TOLERANCE_DEG
        )

    def _lead_clearance_m(self, ego_vehicle) -> float | None:
        """Longitudinal distance to the nearest vehicle directly ahead of ego."""
        world = self.carla_client.get_world()
        if not world:
            return None
        ego_tf = ego_vehicle.get_transform()
        forward = ego_tf.get_forward_vector()
        best: float | None = None
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id == ego_vehicle.id:
                continue
            loc = actor.get_transform().location
            rel_x = loc.x - ego_tf.location.x
            rel_y = loc.y - ego_tf.location.y
            lon = rel_x * forward.x + rel_y * forward.y
            if lon > 0.5:
                best = lon if best is None else min(best, lon)
        return best

    def _centering_creep_active(self, ego_vehicle) -> bool:
        """True while post-merge centering may creep forward (never into a lead vehicle)."""
        if not self._centering_side:
            return False
        errors = self._frozen_centering_errors(ego_vehicle)
        if errors is None or lcc.is_centered_in_target_frame(*errors):
            return False
        lead_m = self._lead_clearance_m(ego_vehicle)
        if lead_m is not None and lead_m < lc.CENTER_CREEP_MIN_LEAD_M:
            return False
        return True

    def _centering_waypoint(self, ego_vehicle):
        """Lane center to steer toward after / while finishing a lane change."""
        base_wp = self._driving_waypoint(ego_vehicle)
        if base_wp is None:
            return None
        candidates = [self._lookahead(base_wp)]
        plan = self._active_centering_plan()
        if (
            plan
            and plan.target_lane_id is not None
            and base_wp.lane_id == plan.target_lane_id
            and plan.source_anchor_x is not None
        ):
            travel = self._merge_travel_m(ego_vehicle, plan)
            tgt_la = self._frozen_lane_lookahead(plan, "target", travel)
            if tgt_la:
                candidates.append(self._lookahead(tgt_la))
        return min(
            candidates,
            key=lambda wp: abs(self._lat_yaw_vs_waypoint(ego_vehicle, wp, wp)[0]),
        )

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
        if not ahead:
            return waypoint
        # next() branches at junctions in arbitrary order; hold the
        # straightest continuation so follow_lane never wanders onto a
        # random turn connector (turns are explicit junction actions).
        return jp.straightest_waypoint(ahead, waypoint.transform.rotation.yaw)

    def _advance_along_lane(self, waypoint, dist_m: float):
        """Advance ``waypoint`` forward ``dist_m`` along its lane, freezing at junctions.

        ``waypoint.next(d)`` returns an arbitrary successor inside a junction, so a
        frozen lane lookahead corrupts the instant the advance crosses a junction
        boundary: the reference jumps onto a curving turn connector (heading and
        position rotate), the merge controller reads a huge lateral/yaw error and
        steers the ego back out of the lane it just merged into. When the anchor is
        on an ordinary road, walk forward in small steps and stop at the last pose
        before the junction. When it already starts inside a junction, fall back to
        next() since there is no pre-junction pose to hold.
        """
        if waypoint is None or dist_m <= 0:
            return waypoint
        if waypoint.is_junction:
            ahead = waypoint.next(dist_m)
            return ahead[0] if ahead else waypoint
        cur = waypoint
        remaining = dist_m
        while remaining > 1e-3:
            step = min(LANE_ADVANCE_STEP_M, remaining)
            ahead = cur.next(step)
            if not ahead:
                break
            cand = ahead[0]
            if cand.is_junction:
                break  # freeze at the last pose before the junction
            cur = cand
            remaining -= step
        return cur

    def _adjacent_lane_waypoint_raw(self, waypoint, side: str):
        """Adjacent driving-lane waypoint at the same longitudinal position."""
        if not waypoint:
            return None
        if side == "left":
            lane_wp = waypoint.get_left_lane()
        else:
            lane_wp = waypoint.get_right_lane()
        if lane_wp is None or lane_wp.lane_type != carla.LaneType.Driving:
            return None
        return lane_wp

    def _adjacent_lane_waypoint(self, waypoint, side: str):
        lane_wp = self._adjacent_lane_waypoint_raw(waypoint, side)
        if lane_wp is None:
            return None
        return self._lookahead(lane_wp)

    def _coerce_lane_id(self, waypoint, desired_lane_id: int | None):
        """Walk laterally on the map until ``waypoint`` matches ``desired_lane_id``."""
        if waypoint is None or desired_lane_id is None:
            return waypoint
        if waypoint.lane_id == desired_lane_id:
            return waypoint
        queue = [waypoint]
        seen: set[tuple[int, int]] = set()
        while queue:
            cur = queue.pop(0)
            key = (cur.road_id, cur.lane_id)
            if key in seen:
                continue
            seen.add(key)
            if cur.lane_id == desired_lane_id:
                return cur
            for getter in (cur.get_left_lane, cur.get_right_lane):
                nxt = getter()
                if nxt is not None and nxt.lane_type == carla.LaneType.Driving:
                    queue.append(nxt)
        return waypoint

    def _waypoint_on_lane_id(self, ego_vehicle, lane_id: int | None):
        """Driving waypoint on ``lane_id`` near ego's along-road position."""
        if lane_id is None:
            return None
        base_wp = self._driving_waypoint(ego_vehicle)
        if base_wp is None:
            return None
        return self._coerce_lane_id(base_wp, lane_id)

    def _frozen_lane_lookahead(self, plan: mp.ManeuverPlan, role: str, travel_m: float):
        """Advance from a frozen merge anchor along the road by ``travel_m``."""
        if role == "source":
            ax, ay, lane_id = (
                plan.source_anchor_x,
                plan.source_anchor_y,
                plan.source_lane_id,
            )
        else:
            ax, ay, lane_id = (
                plan.target_anchor_x,
                plan.target_anchor_y,
                plan.target_lane_id,
            )
        if ax is None or ay is None:
            return None
        world = self.carla_client.get_world()
        if not world:
            return None
        wp = world.get_map().get_waypoint(
            carla.Location(ax, ay, 0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        wp = self._coerce_lane_id(wp, lane_id)
        if wp is None:
            return None
        dist = max(0.0, travel_m) + lc.LOOKAHEAD_M
        return self._advance_along_lane(wp, dist)

    def _ego_speed(self, ego_vehicle) -> float:
        v = ego_vehicle.get_velocity()
        return (v.x**2 + v.y**2 + v.z**2) ** 0.5

    def ego_pose_snapshot(self, ego_vehicle=None) -> dict:
        """Absolute ego pose + map lane, for pre-step / motion-summary logging."""
        ego_vehicle = ego_vehicle or self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            return {}
        tf = ego_vehicle.get_transform()
        wp = self._driving_waypoint(ego_vehicle)
        return {
            "x": round(tf.location.x, 2),
            "y": round(tf.location.y, 2),
            "yaw": round(tf.rotation.yaw, 2),
            "lane_id": wp.lane_id if wp else None,
            "road_id": wp.road_id if wp else None,
            "speed": round(self._ego_speed(ego_vehicle), 2),
        }

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
            control.throttle = min(
                MAX_THROTTLE, max(FOLLOW_ACCEL_MIN_THROTTLE, SPEED_KP * speed_error)
            )
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
        lat_gain: float | None = None,
        yaw_gain: float | None = None,
    ) -> tuple[float, float]:
        """Lateral error vs current lane; yaw error vs lookahead heading."""
        if not lane_wp:
            return 0.0, 0.0

        lat_gain = lat_gain if lat_gain is not None else lc.LAT_GAIN
        yaw_gain = yaw_gain if yaw_gain is not None else lc.YAW_GAIN
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
            lat_gain=lat_gain,
            yaw_gain=yaw_gain,
            max_steer=max_steer,
            lateral_weight=lc.lateral_weight_for_yaw(yaw_err),
        )
        self._last_steer_info = {"yaw_err": yaw_err, "max_steer": max_steer}
        return steer, lat_err

    def _steer_centering_toward_waypoint(
        self,
        ego_vehicle,
        lane_wp,
        heading_wp,
    ) -> tuple[float, float]:
        """Stronger post-merge steer: lateral priority and minimum authority."""
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
        max_steer = lc.speed_scaled_max_steer(speed, lane_change=True)
        steer = lc.compute_centering_steer(
            lat_err,
            yaw_err,
            lat_gain=lc.CENTER_LAT_GAIN,
            yaw_gain=lc.CENTER_YAW_GAIN,
            max_steer=max_steer,
        )
        self._last_steer_info = {"yaw_err": yaw_err, "max_steer": max_steer}
        return steer, lat_err

    def _steer_toward_point(
        self,
        ego_vehicle,
        target_x: float,
        target_y: float,
        right_x: float,
        right_y: float,
        heading_yaw_deg: float,
        *,
        lane_change: bool,
    ) -> tuple[float, float]:
        """Steer toward an arbitrary point (used for the interpolated merge target)."""
        ego_tf = ego_vehicle.get_transform()
        lat_err = lc.lateral_error_m(
            ego_tf.location.x,
            ego_tf.location.y,
            target_x,
            target_y,
            right_x,
            right_y,
        )
        yaw_err = lc.normalize_yaw_error(heading_yaw_deg - ego_tf.rotation.yaw)
        speed = self._ego_speed(ego_vehicle)
        max_steer = lc.speed_scaled_max_steer(speed, lane_change=lane_change)
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
        self._last_steer_info = {"yaw_err": yaw_err, "max_steer": max_steer}
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
            heading_wp = self._centering_waypoint(ego_vehicle)
            if heading_wp:
                return heading_wp
            self._centering_side = None
            self._target_lane_id = None
            self._centering_plan = None

        return self._lookahead(base_wp)

    def _junction_path_steer(self, ego_vehicle) -> tuple[float, float] | None:
        """Steer along the committed junction path.

        Returns None once the path completes so the caller falls back to
        ordinary lane steering (in the exit lane) for the rest of this tick —
        this is what lets follow_lane/yield/stop track a committed turn across
        many steps and then resume plain lane-keeping without a separate
        "completed" flag.
        """
        plan = self._junction_plan
        ego_tf = ego_vehicle.get_transform()
        idx = plan.nearest_index(
            ego_tf.location.x, ego_tf.location.y, self._junction_idx
        )
        self._junction_idx = idx

        if plan.is_complete(ego_tf.location.x, ego_tf.location.y, idx):
            log_stage(
                logger,
                "CARLA",
                "junction %s complete -> exit road=%s lane=%s",
                plan.action,
                plan.exit_road_id,
                plan.exit_lane_id,
            )
            self._junction_plan = None
            self._junction_idx = 0
            return None

        # Cross-track error is the perpendicular distance to the path at the
        # NEAREST pose; heading is fed forward from a lookahead pose so the car
        # anticipates the curve. Measuring lateral error against the lookahead
        # pose (as before) reports ~1.2 m of offset on a junction-radius curve
        # even when perfectly on the path, which permanently trips the
        # "off-path" branch in compute_junction_steer (yaw feed-forward gutted +
        # a phantom lateral correction) and cuts the corner.
        near = plan.poses[idx]
        ref = plan.lookahead_pose(idx)
        lat_err = lc.lateral_error_m(
            ego_tf.location.x,
            ego_tf.location.y,
            near.x,
            near.y,
            near.right_x,
            near.right_y,
        )
        yaw_err = lc.normalize_yaw_error(ref.yaw_deg - ego_tf.rotation.yaw)
        steer = jp.compute_junction_steer(lat_err, yaw_err)
        self._last_steer_info = {"yaw_err": yaw_err, "max_steer": jp.JUNCTION_MAX_STEER}
        return steer, lat_err

    def _apply_junction_speed_cap(self, target_speed: float) -> float:
        """Cap speed while curving through a committed turn; a straight pass
        through a junction drives at the normal follow_lane speed."""
        if self._junction_plan is not None and self._junction_plan.direction != "straight":
            return min(target_speed, jp.JUNCTION_TURN_SPEED_MPS)
        return target_speed

    def _lane_steer(self, ego_vehicle, action: str) -> tuple[float, float]:
        if self._junction_plan is not None:
            result = self._junction_path_steer(ego_vehicle)
            if result is not None:
                return result
            # Plan completed this tick: fall through to ordinary lane steering.

        base_wp = self._driving_waypoint(ego_vehicle)
        if self._centering_side and action in ("follow_lane", "yield", "stop"):
            errors = self._frozen_centering_errors(ego_vehicle)
            if errors is not None:
                lat_err, yaw_err = errors
                speed = self._ego_speed(ego_vehicle)
                max_steer = lcc.lane_change_max_steer(speed)
                steer = lcc.compute_lane_change_steer(
                    lat_err, yaw_err, max_steer=max_steer
                )
                self._last_steer_info = {"yaw_err": yaw_err, "max_steer": max_steer}
                if lcc.is_centered_in_target_frame(lat_err, yaw_err):
                    log_stage(
                        logger,
                        "CARLA",
                        "lane centered action=%s lat_err=%.2fm yaw_err=%.1f°",
                        action,
                        lat_err,
                        yaw_err,
                    )
                    self._clear_maneuver()
                return steer, lat_err

        heading_wp = self._target_waypoint(ego_vehicle, action)
        lane_wp = base_wp
        if action in ("change_lane_left", "change_lane_right", "overtake"):
            lane_wp = heading_wp or base_wp
        steer, lat_err = self._steer_toward_waypoint(
            ego_vehicle,
            lane_wp,
            heading_wp or base_wp,
            lane_change=action
            in ("change_lane_left", "change_lane_right", "overtake"),
        )
        return steer, lat_err

    def _build_lane_change_plan(self, ego_vehicle, action: str) -> mp.ManeuverPlan:
        """Resolve a lane change into a time/distance-parameterized merge plan."""
        current_speed = self._ego_speed(ego_vehicle)
        side = "left" if action in ("change_lane_left", "overtake") else "right"
        base_wp = self._driving_waypoint(ego_vehicle)
        lane_width = base_wp.lane_width if base_wp else mp.DEFAULT_LANE_WIDTH_M
        adj = self._adjacent_lane_waypoint_raw(base_wp, side) if base_wp else None

        duration = mp.merge_duration_s(lane_width, STEP_INTERVAL_S)
        target_speed = mp.merge_target_speed_mps(
            current_speed,
            overtake=(action == "overtake"),
            max_speed_mps=DEFAULT_MAX_SPEED_MPS,
            min_from_rest_mps=MIN_FOLLOW_FROM_REST_MPS,
            overtake_factor=OVERTAKE_SPEED_FACTOR,
            stationary_speed_mps=STATIONARY_SPEED_MPS,
        )
        distance = mp.merge_distance_m(target_speed, duration)

        if adj is None:
            logger.warning("No %s driving lane; falling back to follow_lane.", side)
            return mp.ManeuverPlan(
                action=action,
                side=side,
                is_lane_change=True,
                duration_s=duration,
                distance_m=distance,
                lateral_offset_m=lane_width,
                target_speed_mps=self._follow_lane_target(current_speed),
                start_speed_mps=current_speed,
                is_fallback=True,
            )

        ego_tf = ego_vehicle.get_transform()
        src_loc = base_wp.transform.location
        tgt_loc = adj.transform.location
        right = base_wp.transform.get_right_vector()
        lateral_offset = mp.lateral_spacing_m(
            src_loc.x,
            src_loc.y,
            tgt_loc.x,
            tgt_loc.y,
            right.x,
            right.y,
        )
        if lateral_offset < 0.5:
            lateral_offset = lane_width
        return mp.ManeuverPlan(
            action=action,
            side=side,
            is_lane_change=True,
            duration_s=duration,
            distance_m=distance,
            lateral_offset_m=lateral_offset,
            target_speed_mps=target_speed,
            start_speed_mps=current_speed,
            source_lane_id=base_wp.lane_id if base_wp else None,
            source_road_id=base_wp.road_id if base_wp else None,
            target_lane_id=adj.lane_id,
            source_anchor_x=src_loc.x,
            source_anchor_y=src_loc.y,
            source_anchor_yaw=base_wp.transform.rotation.yaw,
            target_anchor_x=tgt_loc.x,
            target_anchor_y=tgt_loc.y,
            target_anchor_yaw=adj.transform.rotation.yaw,
            start_ego_x=ego_tf.location.x,
            start_ego_y=ego_tf.location.y,
        )

    def _lane_change_control(self, ego_vehicle, action: str, elapsed_s: float):
        """Per-tick control: direct merge to frozen target-lane center."""
        plan = self._plan
        if plan is None:
            plan = self._plan = self._build_lane_change_plan(ego_vehicle, action)
        current_speed = self._ego_speed(ego_vehicle)

        if plan.is_fallback:
            steer, lat_err = self._lane_steer(ego_vehicle, "follow_lane")
            target_speed = self._follow_lane_target(current_speed)
            control = self._drive_with_target_speed(
                ego_vehicle, target_speed, steer, sustain_cruise=True
            )
            return control, target_speed, lat_err

        if (
            plan.source_anchor_x is None
            or plan.start_ego_x is None
            or plan.source_anchor_yaw is None
        ):
            steer, lat_err = self._lane_steer(ego_vehicle, "follow_lane")
            control = self._drive_with_target_speed(
                ego_vehicle, plan.target_speed_mps, steer, sustain_cruise=True
            )
            return control, plan.target_speed_mps, lat_err

        ego_tf = ego_vehicle.get_transform()
        travel_m = mp.forward_travel_m(
            ego_tf.location.x,
            ego_tf.location.y,
            plan.start_ego_x,
            plan.start_ego_y,
            plan.source_anchor_yaw,
        )
        src_la = self._frozen_lane_lookahead(plan, "source", travel_m)
        tgt_la = self._frozen_lane_lookahead(plan, "target", travel_m)
        src_pose = lcc.lane_pose_from_waypoint(src_la)
        tgt_pose = lcc.lane_pose_from_waypoint(tgt_la)

        if src_pose is None or tgt_pose is None:
            steer, lat_err = self._lane_steer(ego_vehicle, "follow_lane")
            control = self._drive_with_target_speed(
                ego_vehicle, plan.target_speed_mps, steer, sustain_cruise=True
            )
            return control, plan.target_speed_mps, lat_err

        lateral_frac = lcc.direct_lateral_fraction(elapsed_s, plan.duration_s)
        steer, lat_err, yaw_err = lcc.steer_toward_frozen_merge(
            ego_tf.location.x,
            ego_tf.location.y,
            ego_tf.rotation.yaw,
            current_speed,
            src_pose,
            tgt_pose,
            lateral_frac,
        )
        max_steer = lcc.lane_change_max_steer(current_speed)
        self._last_steer_info = {"yaw_err": yaw_err, "max_steer": max_steer}

        if lateral_frac >= 1.0 and lcc.is_centered_in_target_frame(lat_err, yaw_err):
            log_stage(
                logger,
                "CARLA",
                "lane centered action=%s lat_err=%.2fm yaw_err=%.1f°",
                action,
                lat_err,
                yaw_err,
            )
            self._centering_side = None
            self._target_lane_id = None

        control = self._drive_with_target_speed(
            ego_vehicle, plan.target_speed_mps, steer
        )
        return control, plan.target_speed_mps, lat_err

    def _junction_hold_stop(self, ego_vehicle):
        """Requested turn is impossible: hold the lane and brake to a stop."""
        steer, lat_err = self._lane_steer(ego_vehicle, "stop")
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = steer
        control.brake = 1.0
        return control, 0.0, lat_err

    def _control_for(self, ego_vehicle, action: str, elapsed_s: float):
        """Compute (control, target_speed, lat_err) for the action at this instant."""
        current_speed = self._ego_speed(ego_vehicle)

        if action in jp.JUNCTION_ACTIONS:
            if self._junction_plan is None and self._junction_action_infeasible:
                return self._junction_hold_stop(ego_vehicle)
            # Round 1 (commit) drives exactly like follow_lane: obstacle-aware
            # speed, steering picked up from the committed path by _lane_steer.
            steer, lat_err = self._lane_steer(ego_vehicle, action)
            target_speed = self._apply_junction_speed_cap(
                self._follow_lane_target(current_speed)
            )
            control = self._drive_with_target_speed(
                ego_vehicle, target_speed, steer, sustain_cruise=True
            )
            return control, target_speed, lat_err

        if action == "stop":
            steer, lat_err = self._lane_steer(ego_vehicle, action)
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = steer
            control.brake = 1.0
            return control, 0.0, lat_err

        if action == "yield":
            steer, lat_err = self._lane_steer(ego_vehicle, action)
            target_speed = self._yield_target(current_speed)
            centering_creep = self._centering_creep_active(ego_vehicle)
            if centering_creep:
                target_speed = max(target_speed, lc.CENTER_CREEP_SPEED_MPS)
            control = self._drive_with_target_speed(
                ego_vehicle,
                target_speed,
                steer,
                brake_only=not centering_creep,
            )
            return control, target_speed, lat_err

        if action == "follow_lane":
            # Round 2: drives the same whether or not a junction plan is
            # committed — _lane_steer/_apply_junction_speed_cap pick it up.
            steer, lat_err = self._lane_steer(ego_vehicle, action)
            target_speed = self._apply_junction_speed_cap(
                self._follow_lane_target(current_speed)
            )
            control = self._drive_with_target_speed(
                ego_vehicle, target_speed, steer, sustain_cruise=True
            )
            return control, target_speed, lat_err

        if action in ("change_lane_left", "change_lane_right", "overtake"):
            return self._lane_change_control(ego_vehicle, action, elapsed_s)

        logger.error("Unhandled action: %s", action)
        return carla.VehicleControl(), 0.0, 0.0

    def _emit_telemetry(self, ego_vehicle, action, control, target_speed, lat_err):
        """Write one full telemetry row and a sampled human-readable traj line."""
        tf = ego_vehicle.get_transform()
        wp = self._driving_waypoint(ego_vehicle)
        lane_id = wp.lane_id if wp else None
        road_id = wp.road_id if wp else None
        speed = self._ego_speed(ego_vehicle)
        frac = (
            round(
                lcc.direct_lateral_fraction(self._elapsed_s, self._plan.duration_s),
                3,
            )
            if (self._plan and self._plan.is_lane_change)
            else None
        )
        info = self._last_steer_info
        row = {
            "step": self._step_index,
            "tick": self._tick_index,
            "sim_time_s": round(self._elapsed_s, 3),
            "action": action,
            "frac": frac,
            "ego_x": round(tf.location.x, 3),
            "ego_y": round(tf.location.y, 3),
            "ego_yaw": round(tf.rotation.yaw, 2),
            "lane_id": lane_id,
            "road_id": road_id,
            "speed_mps": round(speed, 3),
            "target_speed_mps": round(target_speed, 3),
            "lat_err_m": round(lat_err, 3),
            "yaw_err_deg": round(info.get("yaw_err", 0.0), 2),
            "max_steer": round(info.get("max_steer", 0.0), 3),
            "steer": round(control.steer, 3),
            "throttle": round(control.throttle, 3),
            "brake": round(control.brake, 3),
        }
        telemetry.write_row(row)

        lane_changed = (
            self._last_telem_lane_id is not None
            and lane_id is not None
            and lane_id != self._last_telem_lane_id
        )
        self._last_telem_lane_id = lane_id
        total = self._total_ticks if self._total_ticks > 0 else self._tick_index + 1
        if telemetry.should_sample(self._tick_index, total, lane_changed=lane_changed):
            log_stage(
                logger,
                "traj",
                "t=%d sim=%.2fs act=%s frac=%s lane=%s pos=(%.1f,%.1f) yaw=%.1f "
                "spd=%.2f tgt=%.2f lat_err=%.2f yaw_err=%.1f steer=%.3f thr=%.2f brk=%.2f",
                self._tick_index,
                self._elapsed_s,
                action,
                frac if frac is not None else "-",
                lane_id,
                tf.location.x,
                tf.location.y,
                tf.rotation.yaw,
                speed,
                target_speed,
                lat_err,
                info.get("yaw_err", 0.0),
                control.steer,
                control.throttle,
                control.brake,
            )

    def _log_apply(self, action, current_speed, target_speed, control, lat_err):
        if self._plan and self._plan.is_lane_change and not self._plan.is_fallback:
            log_stage(
                logger,
                "CARLA",
                "lane_target side=%s lat_err=%.2fm steer=%.3f merge_d=%.1fm merge_t=%.2fs "
                "lat_offset=%.2fm tgt_speed=%.2f peak_accel=%.2f lane %s->%s",
                self._plan.side,
                lat_err,
                control.steer,
                self._plan.distance_m,
                self._plan.duration_s,
                self._plan.lateral_offset_m,
                self._plan.target_speed_mps,
                mp.peak_lateral_accel_mps2(
                    self._plan.lateral_offset_m, self._plan.duration_s
                ),
                self._plan.source_lane_id,
                self._plan.target_lane_id,
            )
        elif action == "stop" and lat_err != 0.0:
            log_stage(
                logger,
                "CARLA",
                "lane_hold action=stop lat_err=%.2fm steer=%.3f",
                lat_err,
                control.steer,
            )
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

    def begin_action(self, action: str, ego_vehicle=None) -> bool:
        """Commit to an action: build its plan and apply the first control.

        The maneuver is then refined every physics tick by tick().
        """
        ego_vehicle = ego_vehicle or self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not initialized.")
            return False

        self._active_action = action
        self._elapsed_s = 0.0
        if self._centering_side and self._plan and not self._plan.is_fallback:
            self._centering_plan = self._plan
        self._plan = None
        self._merge_logged = False
        current_speed = self._ego_speed(ego_vehicle)

        if action in jp.JUNCTION_ACTIONS:
            # Round-1 commit: build the path once. If already committed to the
            # same direction, keep the existing plan (and its progress index)
            # instead of rebuilding — begin_action can be re-entered for the
            # same step's initial control application.
            self._clear_maneuver()
            direction = jp.ACTION_TO_DIRECTION[action]
            if self._junction_plan is None or self._junction_plan.direction != direction:
                self._junction_plan = jp.build_junction_plan(
                    self._driving_waypoint(ego_vehicle), action
                )
                self._junction_idx = 0
            self._junction_action_infeasible = self._junction_plan is None
            if self._junction_plan is None:
                logger.warning(
                    "No %s connector at the junction ahead; holding a safe stop.",
                    direction,
                )
            else:
                log_stage(
                    logger,
                    "CARLA",
                    "junction committed action=%s dir=%s entry_d=%.1fm path=%d poses "
                    "(%.1fm) exit road=%s lane=%s",
                    action,
                    self._junction_plan.direction,
                    self._junction_plan.junction_distance_m,
                    len(self._junction_plan.poses),
                    self._junction_plan.cum_s[-1],
                    self._junction_plan.exit_road_id,
                    self._junction_plan.exit_lane_id,
                )
        elif action in ("change_lane_left", "change_lane_right", "overtake"):
            # A lateral move changes the source lane, invalidating any committed
            # junction connector (it was frozen from the old lane); the agent
            # re-decides direction (round 1) once the lane change completes.
            if self._junction_plan is not None:
                log_stage(
                    logger,
                    "CARLA",
                    "junction commitment (%s) discarded by %s",
                    self._junction_plan.direction,
                    action,
                )
            self._junction_plan = None
            self._junction_idx = 0
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
            self._plan = self._build_lane_change_plan(ego_vehicle, action)
            if not self._plan.is_fallback:
                self._begin_lane_maneuver(action)
                self._target_lane_id = self._plan.target_lane_id
                self._centering_plan = self._plan
        else:
            # follow_lane / yield / stop (round 2): leave a committed junction
            # plan untouched so these actions keep tracking it across steps.
            self._clear_lateral_tracking()

        control, target_speed, lat_err = self._control_for(ego_vehicle, action, 0.0)
        ego_vehicle.apply_control(control)
        self._tick_index = 0
        self._log_apply(action, current_speed, target_speed, control, lat_err)
        self._emit_telemetry(ego_vehicle, action, control, target_speed, lat_err)
        return True

    def tick(self, dt: float) -> None:
        """Advance the active maneuver one physics sub-step and re-apply control.

        Called once per CARLA tick by the main loop so steering/throttle follow
        the planned trajectory instead of replaying a single frozen control.
        """
        if self._active_action is None:
            return
        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            return

        self._elapsed_s += dt
        self._tick_index += 1
        control, target_speed, lat_err = self._control_for(
            ego_vehicle, self._active_action, self._elapsed_s
        )
        ego_vehicle.apply_control(control)
        self._emit_telemetry(
            ego_vehicle, self._active_action, control, target_speed, lat_err
        )

        if (
            self._plan
            and self._plan.is_lane_change
            and not self._plan.is_fallback
            and not self._merge_logged
            and self._plan.is_time_complete(self._elapsed_s)
        ):
            self._merge_logged = True
            wp = self._driving_waypoint(ego_vehicle)
            ego_lane = wp.lane_id if wp else None
            frozen = self._frozen_centering_errors(ego_vehicle)
            achieved = (
                frozen is not None and lcc.is_centered_in_target_frame(*frozen)
            )
            log_stage(
                logger,
                "CARLA",
                "merge complete action=%s elapsed=%.2fs lat_err=%.2fm "
                "achieved=%s ego_lane=%s target_lane=%s yaw_err=%.1f°",
                self._active_action,
                self._elapsed_s,
                lat_err,
                achieved,
                ego_lane,
                self._plan.target_lane_id,
                self._last_steer_info.get("yaw_err", 0.0),
            )
            if not achieved:
                logger.warning(
                    "%s [CARLA] lane change NOT centered (ego_lane=%s target=%s "
                    "lat_err=%.2fm yaw_err=%.1f°); keeping centering active",
                    "[pipeline]",
                    ego_lane,
                    self._plan.target_lane_id,
                    lat_err,
                    self._last_steer_info.get("yaw_err", 0.0),
                )

    def describe_action(self, action: str) -> dict:
        """Resolve an action into concrete waypoint/merge geometry (for preview).

        Uses the same maneuver_planner math as execution so the preview the LLM
        sees matches what will actually be driven.
        """
        if action not in self.valid_actions:
            return {"action": action, "error": f"Unknown action: {action}"}

        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            return {"action": action, "error": "Ego vehicle not initialized."}

        current_speed = self._ego_speed(ego_vehicle)
        base_wp = self._driving_waypoint(ego_vehicle)
        lane_width = base_wp.lane_width if base_wp else mp.DEFAULT_LANE_WIDTH_M
        info: dict = {
            "action": action,
            "ego_speed_mps": round(current_speed, 2),
            "current_lane_id": base_wp.lane_id if base_wp else None,
            "road_id": base_wp.road_id if base_wp else None,
            "lane_width_m": round(lane_width, 2),
        }

        if action in ("change_lane_left", "change_lane_right", "overtake"):
            side = "left" if action in ("change_lane_left", "overtake") else "right"
            adj = self._adjacent_lane_waypoint_raw(base_wp, side) if base_wp else None
            duration = mp.merge_duration_s(lane_width, STEP_INTERVAL_S)
            target_speed = mp.merge_target_speed_mps(
                current_speed,
                overtake=(action == "overtake"),
                max_speed_mps=DEFAULT_MAX_SPEED_MPS,
                min_from_rest_mps=MIN_FOLLOW_FROM_REST_MPS,
                overtake_factor=OVERTAKE_SPEED_FACTOR,
                stationary_speed_mps=STATIONARY_SPEED_MPS,
            )
            distance = mp.merge_distance_m(target_speed, duration)
            info.update(
                {
                    "kind": "lane_change",
                    "target_side": side,
                    "target_lane_available": adj is not None,
                    "target_lane_id": adj.lane_id if adj else None,
                    "lateral_offset_m": round(lane_width, 2),
                    "merge_distance_m": round(distance, 2),
                    "merge_duration_s": round(duration, 2),
                    "merge_settling_time_s": round(
                        mp.merge_settling_time_s(STEP_INTERVAL_S), 2
                    ),
                    "target_speed_mps": round(target_speed, 2),
                    "peak_lateral_accel_mps2": round(
                        mp.peak_lateral_accel_mps2(lane_width, duration), 2
                    ),
                }
            )
            if adj is not None:
                la = self._lookahead(adj)
                info["target_waypoint"] = {
                    "x": round(la.transform.location.x, 2),
                    "y": round(la.transform.location.y, 2),
                }
        elif action in jp.JUNCTION_ACTIONS:
            direction = jp.ACTION_TO_DIRECTION[action]
            scan = jp.scan_ahead(base_wp) if base_wp else jp.JunctionScan()
            options = scan.options
            branch = scan.branches.get(direction)
            info.update(
                {
                    "kind": "junction_turn",
                    "direction": direction,
                    "junction_ahead": scan.kind == "junction",
                    "junction_kind": scan.kind,
                    "junction_distance_m": scan.distance_m,
                    "junction_options": options,
                    "junction_preferred_action": (
                        jp.preferred_junction_action(options)
                        if scan.kind == "junction"
                        else None
                    ),
                    "option_available": bool(branch),
                    "target_speed_mps": round(jp.JUNCTION_TURN_SPEED_MPS, 2),
                    "already_committed": (
                        self._junction_plan is not None
                        and self._junction_plan.direction == direction
                    ),
                    "committed_direction": (
                        self._junction_plan.direction if self._junction_plan else None
                    ),
                }
            )
            if branch:
                exit_wp = branch[-1]
                info.update(
                    {
                        "exit_road_id": exit_wp.road_id,
                        "exit_lane_id": exit_wp.lane_id,
                        "exit_yaw": round(exit_wp.transform.rotation.yaw, 2),
                        "exit_waypoint": {
                            "x": round(exit_wp.transform.location.x, 2),
                            "y": round(exit_wp.transform.location.y, 2),
                        },
                    }
                )
        elif action == "follow_lane":
            info.update(
                {
                    "kind": "lane_keep",
                    "lookahead_m": round(lc.LOOKAHEAD_M, 2),
                    "target_speed_mps": round(
                        self._follow_lane_target(current_speed), 2
                    ),
                }
            )
            if base_wp is not None:
                la = self._lookahead(base_wp)
                info["target_waypoint"] = {
                    "x": round(la.transform.location.x, 2),
                    "y": round(la.transform.location.y, 2),
                }
        elif action == "yield":
            info.update(
                {
                    "kind": "slow",
                    "target_speed_mps": round(self._yield_target(current_speed), 2),
                }
            )
        elif action == "stop":
            info.update({"kind": "halt", "target_speed_mps": 0.0})

        return info

    def execute_action(self, action: str):
        if action not in self.valid_actions:
            logger.error("Invalid action: %s", action)
            return False

        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not initialized.")
            return False

        return self.begin_action(action, ego_vehicle)
