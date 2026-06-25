"""Frozen-frame lane change geometry and steering.

Lane changes steer toward a merge point interpolated between frozen source/target
lane anchors advanced by along-road travel.  Lateral error and heading are always
measured in the **target-lane frame** so junction lane_id remaps cannot flip the
error signal mid-maneuver.
"""
from __future__ import annotations

from dataclasses import dataclass

from simulation import lane_controller as lc
from simulation import maneuver_planner as mp
# Aggressive direct lane-change steering (separate from post-yield centering).
LANE_CHANGE_LAT_GAIN = float(
    __import__("os").getenv("LANE_CHANGE_LAT_GAIN", "0.085")
)
LANE_CHANGE_YAW_GAIN = float(
    __import__("os").getenv("LANE_CHANGE_YAW_GAIN", "0.022")
)
LANE_CHANGE_MIN_STEER = float(
    __import__("os").getenv("LANE_CHANGE_MIN_STEER", "0.08")
)
LANE_CHANGE_YAW_WEIGHT_OFF_CENTER = float(
    __import__("os").getenv("LANE_CHANGE_YAW_WEIGHT_OFF_CENTER", "0.12")
)
LANE_CHANGE_STEER_FLOOR = float(
    __import__("os").getenv("LANE_CHANGE_STEER_FLOOR", "0.35")
)
LANE_CHANGE_LEAD_MARGIN_M = float(
    __import__("os").getenv("LANE_CHANGE_LEAD_MARGIN_M", "4.0")
)
# Post-merge: enforce lateral correction before the full center tolerance.
POST_MERGE_LAT_PRIORITY_M = float(
    __import__("os").getenv("LANE_CHANGE_POST_MERGE_LAT_PRIORITY_M", "0.35")
)


@dataclass(frozen=True)
class LanePose:
    """A point on a frozen lane centerline with heading and right vector."""

    x: float
    y: float
    yaw_deg: float
    right_x: float
    right_y: float


def lane_pose_from_waypoint(waypoint) -> LanePose | None:
    if waypoint is None:
        return None
    tf = waypoint.transform
    right = tf.get_right_vector()
    return LanePose(
        tf.location.x,
        tf.location.y,
        tf.rotation.yaw,
        right.x,
        right.y,
    )


def direct_lateral_fraction(elapsed_s: float, duration_s: float) -> float:
    """Linear 0→1 profile: direct move to the new lane (no slow S-curve tail)."""
    if duration_s <= 0:
        return 1.0
    return max(0.0, min(1.0, elapsed_s / duration_s))


def merge_target_xy(
    src: LanePose,
    tgt: LanePose,
    lateral_frac: float,
) -> tuple[float, float]:
    return mp.merge_lateral_target_interpolated(
        src.x, src.y, tgt.x, tgt.y, lateral_frac
    )


def errors_in_target_frame(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ref: LanePose,
) -> tuple[float, float]:
    """Lateral + yaw error vs a frozen target-lane pose."""
    lat_err = lc.lateral_error_m(
        ego_x, ego_y, ref.x, ref.y, ref.right_x, ref.right_y
    )
    yaw_err = lc.normalize_yaw_error(ref.yaw_deg - ego_yaw)
    if abs(yaw_err) > 45.0:
        yaw_err = max(-45.0, min(45.0, yaw_err))
    return lat_err, yaw_err


def lane_change_max_steer(speed_mps: float) -> float:
    """Full lane-change authority — do not throttle steer mid-merge."""
    scaled = lc.speed_scaled_max_steer(speed_mps, lane_change=True)
    return min(lc.MAX_STEER_LANE_CHANGE, max(LANE_CHANGE_STEER_FLOOR, scaled))


def compute_lane_change_steer(
    lat_err: float,
    yaw_err: float,
    *,
    max_steer: float,
) -> float:
    """Direct, lateral-first steering for a committed lane change."""
    yaw_err = lc.normalize_yaw_error(yaw_err)
    if abs(yaw_err) > 45.0:
        yaw_err = max(-45.0, min(45.0, yaw_err))
    off_lat = abs(lat_err) > lc.CENTER_TOLERANCE_M
    yaw_weight = LANE_CHANGE_YAW_WEIGHT_OFF_CENTER if off_lat else 1.0
    steer = (-LANE_CHANGE_LAT_GAIN * lat_err) + (
        LANE_CHANGE_YAW_GAIN * yaw_err * yaw_weight
    )
    if off_lat:
        sign = lc.centering_steer_sign(lat_err)
        floor = min(
            max_steer,
            max(LANE_CHANGE_MIN_STEER, abs(lat_err) * LANE_CHANGE_LAT_GAIN * 0.85),
        )
        if abs(steer) < floor or steer * sign < 0:
            steer = sign * floor
    return max(-max_steer, min(max_steer, steer))


def compute_post_merge_centering_steer(
    lat_err: float,
    yaw_err: float,
    *,
    max_steer: float,
) -> float:
    """Settling steer after the timed lateral profile completes.

    Unlike mid-merge steering, yaw is never damped here and lateral sign always
    wins when off-line so a small heading error cannot steer deeper into an
    overshoot (the right-lane failure mode in debug2506_012 step 3).
    """
    yaw_err = lc.normalize_yaw_error(yaw_err)
    if abs(yaw_err) > 45.0:
        yaw_err = max(-45.0, min(45.0, yaw_err))
    steer = (-LANE_CHANGE_LAT_GAIN * lat_err) + (
        LANE_CHANGE_YAW_GAIN * yaw_err
    )
    if abs(lat_err) > POST_MERGE_LAT_PRIORITY_M:
        sign = lc.centering_steer_sign(lat_err)
        floor = min(
            max_steer,
            max(
                LANE_CHANGE_MIN_STEER * 1.75,
                abs(lat_err) * LANE_CHANGE_LAT_GAIN * 3.0,
            ),
        )
        if abs(steer) < floor or steer * sign < 0:
            steer = sign * floor
    return max(-max_steer, min(max_steer, steer))


def is_centered_in_target_frame(
    lat_err: float,
    yaw_err: float,
) -> bool:
    return (
        abs(lat_err) <= lc.CENTER_TOLERANCE_M
        and abs(yaw_err) <= lc.CENTER_YAW_TOLERANCE_DEG
    )


def steer_toward_frozen_merge(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_speed: float,
    src: LanePose,
    tgt: LanePose,
    lateral_frac: float,
) -> tuple[float, float, float]:
    """Return (steer, lat_err, yaw_err) for one lane-change tick."""
    tx, ty = merge_target_xy(src, tgt, lateral_frac)
    heading_yaw = mp.blend_yaw_deg(src.yaw_deg, tgt.yaw_deg, lateral_frac)

    if lateral_frac >= 1.0:
        ref = tgt
    else:
        ref = LanePose(tx, ty, heading_yaw, tgt.right_x, tgt.right_y)

    lat_err, yaw_err = errors_in_target_frame(ego_x, ego_y, ego_yaw, ref)
    max_steer = lane_change_max_steer(ego_speed)
    if lateral_frac >= 1.0:
        steer = compute_post_merge_centering_steer(
            lat_err, yaw_err, max_steer=max_steer
        )
    else:
        steer = compute_lane_change_steer(
            lat_err, yaw_err, max_steer=max_steer
        )
    return steer, lat_err, yaw_err


def merge_lead_ok(merge_distance_m: float, lead_distance_m: float | None) -> bool:
    if lead_distance_m is None:
        return True
    return merge_distance_m + LANE_CHANGE_LEAD_MARGIN_M <= lead_distance_m
