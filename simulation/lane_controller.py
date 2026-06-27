"""Pure helpers for waypoint-based lane keeping and lane-change centering."""
from __future__ import annotations

import os

LOOKAHEAD_M = float(os.getenv("LANE_LOOKAHEAD_M", "6.0"))
LAT_GAIN = float(os.getenv("LANE_LAT_GAIN", "0.028"))
YAW_GAIN = float(os.getenv("LANE_YAW_GAIN", "0.014"))
CENTER_TOLERANCE_M = float(os.getenv("LANE_CENTER_TOLERANCE_M", "0.7"))
CENTER_YAW_TOLERANCE_DEG = float(os.getenv("LANE_CENTER_YAW_TOLERANCE_DEG", "2.0"))
CENTER_LAT_GAIN = float(os.getenv("LANE_CENTER_LAT_GAIN", "0.045"))
CENTER_YAW_GAIN = float(os.getenv("LANE_CENTER_YAW_GAIN", "0.028"))
CENTER_MIN_STEER = float(os.getenv("LANE_CENTER_MIN_STEER", "0.05"))
CENTER_CREEP_SPEED_MPS = float(os.getenv("LANE_CENTER_CREEP_SPEED_MPS", "1.2"))
CENTER_CREEP_MIN_LEAD_M = float(os.getenv("LANE_CENTER_CREEP_MIN_LEAD_M", "10.0"))
CENTER_YAW_WEIGHT_WHILE_OFF_CENTER = float(
    os.getenv("LANE_CENTER_YAW_WEIGHT_OFF_CENTER", "0.35")
)
MAX_STEER_FOLLOW = float(os.getenv("MAX_STEER_FOLLOW", "0.18"))
MAX_STEER_LANE_CHANGE = float(os.getenv("MAX_STEER_LANE_CHANGE", "0.45"))
MAX_STEER_FOLLOW_FLOOR = float(os.getenv("MAX_STEER_FOLLOW_FLOOR", "0.03"))
MAX_STEER_FOLLOW_SCALE = float(os.getenv("MAX_STEER_FOLLOW_SCALE", "0.012"))
MAX_STEER_LANE_CHANGE_FLOOR = float(os.getenv("MAX_STEER_LANE_CHANGE_FLOOR", "0.12"))
MAX_STEER_LANE_CHANGE_SCALE = float(os.getenv("MAX_STEER_LANE_CHANGE_SCALE", "0.03"))


def normalize_yaw_error(yaw_error_deg: float) -> float:
    while yaw_error_deg > 180.0:
        yaw_error_deg -= 360.0
    while yaw_error_deg < -180.0:
        yaw_error_deg += 360.0
    return yaw_error_deg


def lateral_error_m(
    ego_x: float,
    ego_y: float,
    target_x: float,
    target_y: float,
    right_x: float,
    right_y: float,
) -> float:
    """Positive = ego is right of the target lane center (needs left steer)."""
    dx = ego_x - target_x
    dy = ego_y - target_y
    return dx * right_x + dy * right_y


def compute_steer(
    lateral_error_m: float,
    yaw_error_deg: float,
    *,
    lat_gain: float,
    yaw_gain: float,
    max_steer: float,
    max_yaw_deg: float = 25.0,
    lateral_weight: float = 1.0,
) -> float:
    yaw_error_deg = normalize_yaw_error(yaw_error_deg)
    yaw_error_deg = max(-max_yaw_deg, min(max_yaw_deg, yaw_error_deg))
    steer = (-lat_gain * lateral_error_m * lateral_weight) + (yaw_gain * yaw_error_deg)
    return max(-max_steer, min(max_steer, steer))


def centering_steer_sign(lateral_error_m: float) -> float:
    """Steer sign that reduces lateral error (negative lat -> positive steer)."""
    if abs(lateral_error_m) < 1e-6:
        return 0.0
    return -1.0 if lateral_error_m > 0 else 1.0


def compute_centering_steer(
    lateral_error_m: float,
    yaw_error_deg: float,
    *,
    lat_gain: float,
    yaw_gain: float,
    max_steer: float,
    lat_tolerance_m: float = CENTER_TOLERANCE_M,
    min_steer: float = CENTER_MIN_STEER,
    yaw_weight_off_center: float = CENTER_YAW_WEIGHT_WHILE_OFF_CENTER,
) -> float:
    """Post-merge centering: lateral priority and minimum steer when off-line."""
    yaw_error_deg = normalize_yaw_error(yaw_error_deg)
    off_lat = abs(lateral_error_m) > lat_tolerance_m
    yaw_weight = yaw_weight_off_center if off_lat else 1.0
    steer = (-lat_gain * lateral_error_m) + (yaw_gain * yaw_error_deg * yaw_weight)
    if off_lat:
        desired_sign = centering_steer_sign(lateral_error_m)
        scaled_min = min(
            max_steer,
            max(min_steer, abs(lateral_error_m) * lat_gain * 0.65),
        )
        if abs(steer) < scaled_min or steer * desired_sign < 0:
            steer = desired_sign * scaled_min
    return max(-max_steer, min(max_steer, steer))


def lateral_weight_for_yaw(yaw_error_deg: float) -> float:
    """Reduce lateral P-term when heading is badly misaligned (avoids dive off-road)."""
    yaw_error_deg = abs(normalize_yaw_error(yaw_error_deg))
    if yaw_error_deg > 40.0:
        return 0.0
    if yaw_error_deg > 22.0:
        return 0.35
    return 1.0


def speed_scaled_max_steer(ego_speed_m_s: float, *, lane_change: bool) -> float:
    if lane_change:
        cap = MAX_STEER_LANE_CHANGE
        floor = MAX_STEER_LANE_CHANGE_FLOOR
        scale = MAX_STEER_LANE_CHANGE_SCALE
    else:
        cap = MAX_STEER_FOLLOW
        floor = MAX_STEER_FOLLOW_FLOOR
        scale = MAX_STEER_FOLLOW_SCALE
    return min(cap, floor + ego_speed_m_s * scale)

