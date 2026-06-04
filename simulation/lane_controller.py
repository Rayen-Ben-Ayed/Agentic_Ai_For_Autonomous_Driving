"""Pure helpers for waypoint-based lane keeping and lane-change centering."""
from __future__ import annotations

import os


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


def lateral_weight_for_yaw(yaw_error_deg: float) -> float:
    """Reduce lateral P-term when heading is badly misaligned (avoids dive off-road)."""
    yaw_error_deg = abs(normalize_yaw_error(yaw_error_deg))
    if yaw_error_deg > 40.0:
        return 0.0
    if yaw_error_deg > 22.0:
        return 0.35
    return 1.0


def speed_scaled_max_steer(ego_speed_m_s: float, *, lane_change: bool) -> float:
    cap = MAX_STEER_LANE_CHANGE if lane_change else MAX_STEER_FOLLOW
    floor = 0.04 if lane_change else 0.03
    scale = 0.016 if lane_change else 0.012
    return min(cap, floor + ego_speed_m_s * scale)


LOOKAHEAD_M = float(os.getenv("LANE_LOOKAHEAD_M", "6.0"))
LAT_GAIN = float(os.getenv("LANE_LAT_GAIN", "0.028"))
YAW_GAIN = float(os.getenv("LANE_YAW_GAIN", "0.014"))
CENTER_TOLERANCE_M = float(os.getenv("LANE_CENTER_TOLERANCE_M", "0.7"))
MAX_STEER_FOLLOW = float(os.getenv("MAX_STEER_FOLLOW", "0.18"))
MAX_STEER_LANE_CHANGE = float(os.getenv("MAX_STEER_LANE_CHANGE", "0.45"))
