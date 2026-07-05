"""
Predict whether a pedestrian will enter the ego driving corridor from speed and pose.
"""
from __future__ import annotations

import math
import os
from typing import Any

from simulation.maneuver_policy import LANE_CHANGE_MIN_DIST_M, MIN_PLANNING_SPEED_MPS
from simulation.timing_config import AGENT_LATENCY_S, STEP_INTERVAL_S

LANE_HALF_WIDTH_M = float(os.getenv("PED_LANE_HALF_WIDTH_M", "2.5"))
# Half-width of the whole carriageway the ego may occupy (covers both driving
# lanes + margin). A pedestrian crossing anywhere inside this corridor sweeps
# across the ego path regardless of which lane the ego is in, so we react to the
# crossing itself, not only entry into the ego's current lane.
ROAD_HALF_WIDTH_M = float(os.getenv("PED_ROAD_HALF_WIDTH_M", "7.0"))
PED_MAX_LONGITUDINAL_M = float(os.getenv("PED_CONFLICT_MAX_LONGITUDINAL_M", "70.0"))
# A crossing pedestrian is only worth reacting to within this range ahead.
PED_CROSSING_MAX_LONGITUDINAL_M = float(
    os.getenv("PED_CROSSING_MAX_LONGITUDINAL_M", "45.0")
)
# Look-ahead for "will this walker be on the carriageway soon" — one commit
# window plus margin, since each action is committed for a full step blind.
PED_CROSSING_HORIZON_S = float(
    os.getenv("PED_CROSSING_HORIZON_S", str(STEP_INTERVAL_S + 2.0))
)
MIN_LATERAL_SPEED_MPS = float(os.getenv("PED_MIN_LATERAL_SPEED_MPS", "0.35"))
MIN_WALK_SPEED_MPS = float(os.getenv("PED_MIN_WALK_SPEED_MPS", "0.4"))
PED_STOP_DISTANCE_M = float(os.getenv("PED_STOP_DISTANCE_M", "8.0"))
PED_INTENT_YAW_TOLERANCE_DEG = float(os.getenv("PED_INTENT_YAW_TOLERANCE_DEG", "55.0"))


def _is_walker(type_id: str) -> bool:
    return type_id.startswith("walker.")


def _distance_to_lane_edge(lateral_m: float, lane_half_width_m: float) -> float:
    if abs(lateral_m) <= lane_half_width_m:
        return 0.0
    return abs(lateral_m) - lane_half_width_m


def _lateral_closing_rate(lateral_m: float, lateral_vel_mps: float) -> float:
    """Positive when the actor moves toward the lane center (lateral=0)."""
    if abs(lateral_m) <= LANE_HALF_WIDTH_M:
        return 0.0
    if lateral_m > 0.0:
        return max(0.0, -lateral_vel_mps)
    return max(0.0, lateral_vel_mps)


def _closing_toward_center(lateral_m: float, lateral_vel_mps: float) -> float:
    """Speed at which the actor reduces its lateral offset, at any distance.

    Unlike ``_lateral_closing_rate`` this is not gated on being outside the ego
    lane, so it detects a walker that is actively crossing the carriageway.
    """
    if lateral_m > 0.0:
        return max(0.0, -lateral_vel_mps)
    if lateral_m < 0.0:
        return max(0.0, lateral_vel_mps)
    return abs(lateral_vel_mps)


def _heading_toward_lane_center(
    lateral_m: float,
    yaw_deg: float,
    ego_yaw_deg: float,
    *,
    lane_half_width_m: float,
) -> bool:
    """True when body yaw points roughly toward the driving corridor."""
    if abs(lateral_m) <= lane_half_width_m:
        return True
    rel_yaw = math.radians(_normalize_angle_deg(yaw_deg - ego_yaw_deg))
    facing_y = math.sin(rel_yaw)
    if lateral_m > 0.0:
        toward_center_y = -1.0
    else:
        toward_center_y = 1.0
    dot = facing_y * toward_center_y
    return dot >= math.cos(math.radians(PED_INTENT_YAW_TOLERANCE_DEG))


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _prediction_horizon_s() -> float:
    return AGENT_LATENCY_S + STEP_INTERVAL_S


def evaluate_pedestrian_conflict(
    *,
    longitudinal_m: float,
    lateral_m: float,
    lateral_vel_mps: float,
    longitudinal_vel_mps: float,
    speed_mps: float,
    yaw_deg: float,
    ego_yaw_deg: float,
    lane_half_width_m: float = LANE_HALF_WIDTH_M,
    road_half_width_m: float = ROAD_HALF_WIDTH_M,
    max_longitudinal_m: float = PED_MAX_LONGITUDINAL_M,
    crossing_max_longitudinal_m: float = PED_CROSSING_MAX_LONGITUDINAL_M,
    crossing_horizon_s: float = PED_CROSSING_HORIZON_S,
    prediction_horizon_s: float | None = None,
) -> dict[str, Any]:
    """Assess one pedestrian relative to the ego driving corridor."""
    prediction_horizon_s = (
        _prediction_horizon_s() if prediction_horizon_s is None else prediction_horizon_s
    )
    in_lane = abs(lateral_m) <= lane_half_width_m
    ahead = 0.0 < longitudinal_m < max_longitudinal_m
    closing = _lateral_closing_rate(lateral_m, lateral_vel_mps)
    distance_to_lane = _distance_to_lane_edge(lateral_m, lane_half_width_m)

    velocity_intent = closing >= MIN_LATERAL_SPEED_MPS
    heading_intent = (
        speed_mps >= MIN_WALK_SPEED_MPS
        and _heading_toward_lane_center(
            lateral_m, yaw_deg, ego_yaw_deg, lane_half_width_m=lane_half_width_m
        )
    )
    moving_toward_lane = velocity_intent or (
        not velocity_intent and heading_intent and distance_to_lane > 0.0
    )

    time_to_lane_entry_s = None
    if distance_to_lane > 0.0 and closing >= MIN_LATERAL_SPEED_MPS:
        time_to_lane_entry_s = distance_to_lane / closing

    will_enter_lane = in_lane
    predicted = False
    if not in_lane and ahead and moving_toward_lane and time_to_lane_entry_s is not None:
        if time_to_lane_entry_s <= prediction_horizon_s:
            will_enter_lane = True
            predicted = True

    # Road-level crossing: a walker actively moving across the carriageway ahead
    # will sweep through the ego path whichever lane it takes. Flag it as soon as
    # it is (or will soon be) inside the road corridor — not only the ego lane.
    crossing_ahead = 0.0 < longitudinal_m < crossing_max_longitudinal_m
    within_road = abs(lateral_m) <= road_half_width_m
    road_closing = _closing_toward_center(lateral_m, lateral_vel_mps)
    is_crossing = road_closing >= MIN_LATERAL_SPEED_MPS
    distance_to_road = _distance_to_lane_edge(lateral_m, road_half_width_m)
    time_to_road_entry_s = None
    if distance_to_road > 0.0 and is_crossing:
        time_to_road_entry_s = distance_to_road / road_closing

    will_enter_road = within_road
    if not within_road and time_to_road_entry_s is not None:
        will_enter_road = time_to_road_entry_s <= crossing_horizon_s
    crossing_road = (
        crossing_ahead and is_crossing and will_enter_road and not will_enter_lane
    )

    conflict = (ahead and will_enter_lane) or crossing_road
    predicted = predicted or crossing_road
    predicted_longitudinal_m = longitudinal_m
    if time_to_lane_entry_s is not None and not in_lane:
        predicted_longitudinal_m = longitudinal_m + longitudinal_vel_mps * time_to_lane_entry_s

    return {
        "conflict": conflict,
        "predicted": predicted,
        "in_lane": in_lane,
        "crossing_road": crossing_road,
        "longitudinal_m": round(longitudinal_m, 2),
        "lateral_m": round(lateral_m, 2),
        "lateral_vel_mps": round(lateral_vel_mps, 2),
        "distance_to_lane_m": round(distance_to_lane, 2),
        "time_to_lane_entry_s": (
            round(time_to_lane_entry_s, 2) if time_to_lane_entry_s is not None else None
        ),
        "predicted_longitudinal_m": round(predicted_longitudinal_m, 2),
        "moving_toward_lane": moving_toward_lane,
    }


def choose_pedestrian_caution_action(
    *,
    effective_closest_m: float,
    ego_speed_mps: float,
    follow_safe_distance_m: float,
    time_to_lane_entry_s: float | None,
    in_lane: bool,
) -> str:
    """
    Pick yield vs stop when follow_lane is no longer allowed.

    Stop when the pedestrian is very close, already in-lane near the ego, or
    will enter the corridor before the ego can pass safely.
    """
    planning_speed = max(ego_speed_mps, MIN_PLANNING_SPEED_MPS)
    ego_time_to_actor_s = effective_closest_m / planning_speed

    if effective_closest_m <= PED_STOP_DISTANCE_M:
        return "stop"
    if in_lane and effective_closest_m < LANE_CHANGE_MIN_DIST_M:
        return "stop"
    if (
        time_to_lane_entry_s is not None
        and time_to_lane_entry_s <= ego_time_to_actor_s + 0.75
    ):
        return "stop"
    if effective_closest_m < follow_safe_distance_m * 0.45:
        return "stop"
    return "yield"


def assess_nearest_pedestrian_conflict(
    actors: list[dict],
    *,
    ego_yaw_deg: float,
    ego_speed_mps: float,
    follow_safe_distance_m: float = 0.0,
    too_close_for_follow_lane: bool = False,
) -> dict[str, Any]:
    """Scan walkers and return the closest predicted/in-lane conflict."""
    del follow_safe_distance_m, too_close_for_follow_lane, ego_speed_mps
    best: dict[str, Any] | None = None
    best_distance: float | None = None

    for actor in actors:
        if not _is_walker(actor.get("type", "")):
            continue
        ef = actor.get("ego_frame") or {}
        longitudinal = ef.get("longitudinal_m")
        lateral = ef.get("lateral_m")
        if longitudinal is None or lateral is None:
            continue

        assessment = evaluate_pedestrian_conflict(
            longitudinal_m=longitudinal,
            lateral_m=lateral,
            lateral_vel_mps=float(ef.get("lateral_vel_mps") or 0.0),
            longitudinal_vel_mps=float(ef.get("longitudinal_vel_mps") or 0.0),
            speed_mps=float(actor.get("speed") or 0.0),
            yaw_deg=float((actor.get("rotation") or {}).get("yaw") or 0.0),
            ego_yaw_deg=ego_yaw_deg,
        )
        actor["pedestrian_assessment"] = assessment

        if not assessment["conflict"]:
            continue

        distance = float(assessment["predicted_longitudinal_m"])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = {
                "actor_id": actor.get("id"),
                "type": actor.get("type"),
                **assessment,
            }

    return {
        "pedestrian_conflict_ahead": best is not None,
        "pedestrian_conflict_predicted": bool(best and best.get("predicted")),
        "pedestrian_conflict": best,
        "closest_pedestrian_conflict_m": (
            round(best_distance, 2) if best_distance is not None else None
        ),
    }
