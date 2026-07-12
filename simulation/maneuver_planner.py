"""Map- and speed-relative maneuver geometry.

Single source of truth for "what an action means as a trajectory": the
distance/time/lateral offset needed to complete a lane change, the smooth
S-curve lateral profile, and the speed held during the move. Both the action
executor (which drives CARLA per physics tick) and the MCP ``preview_action``
tool import this module so the preview the LLM sees matches what is executed.

These helpers are intentionally pure (no CARLA imports) so they can be unit
tested without a running simulator.
"""
from __future__ import annotations

import math
import os
from typing import Optional
from dataclasses import dataclass
from simulation.timing_config import STEP_INTERVAL_S

# Comfortable lane-change duration target (seconds) before window clamping.
LANE_CHANGE_DURATION_S = float(os.getenv("LANE_CHANGE_DURATION_S", "3.5"))
# Comfortable peak lateral acceleration during a merge (m/s^2).
COMFORT_LATERAL_ACCEL_MPS2 = float(os.getenv("COMFORT_LATERAL_ACCEL_MPS2", "1.5"))
# Lane change lateral profile must reach 100% within this fraction of the agent
# decision window (STEP_INTERVAL_S). The remaining window is for in-step settling.
MERGE_WINDOW_FRACTION = float(os.getenv("MERGE_WINDOW_FRACTION", "0.7"))
# Lower bound on the longitudinal distance a merge consumes (avoids a near-zero
# merge distance when the ego is nearly stopped).
MIN_MERGE_DISTANCE_M = float(os.getenv("MIN_MERGE_DISTANCE_M", "8.0"))
# Fallback lane width when CARLA cannot supply one.
DEFAULT_LANE_WIDTH_M = float(os.getenv("DEFAULT_LANE_WIDTH_M", "3.5"))
# Minimum forward speed held during a merge so the (gentle) steering still yields
# a full lane width within merge_distance_m. A near-stopped ego cannot translate
# steering into lateral travel, so merges crept without clearing the lane.
MERGE_MIN_SPEED_MPS = float(os.getenv("MERGE_MIN_SPEED_MPS", "3.0"))

# Peak of |s''(p)| for the smoothstep profile s(p)=3p^2-2p^3 is 6 (at p=0,1).
_SMOOTHSTEP_ACCEL_PEAK = 6.0


def smoothstep(progress: float) -> float:
    """Clamped cubic smoothstep 3p^2-2p^3 with zero slope at both ends."""
    p = max(0.0, min(1.0, progress))
    return p * p * (3.0 - 2.0 * p)


def lateral_fraction(progress: float) -> float:
    """Fraction of the lateral lane offset that should be covered by ``progress``."""
    return smoothstep(progress)


def _min_duration_for_comfort(lane_width_m: float) -> float:
    """Shortest merge time keeping peak lateral accel within comfort.

    For y(t)=A*s(t/T), peak |lateral accel| = (peak|s''|)*A/T^2, so
    T >= sqrt(peak * A / a_comfort).
    """
    if COMFORT_LATERAL_ACCEL_MPS2 <= 0:
        return LANE_CHANGE_DURATION_S
    return math.sqrt(
        _SMOOTHSTEP_ACCEL_PEAK * lane_width_m / COMFORT_LATERAL_ACCEL_MPS2
    )


def merge_duration_s(
    lane_width_m: float = DEFAULT_LANE_WIDTH_M,
    window_s: float = STEP_INTERVAL_S,
) -> float:
    """Hard deadline for the lateral merge profile within one decision step.

    Always ``MERGE_WINDOW_FRACTION * window_s`` so the S-curve completes in the
    first 70% (default) of STEP_INTERVAL_S regardless of speed or comfort limits.
    The remaining step time is used for in-step lane settling / centering.
    """
    _ = lane_width_m  # kept for call-site compatibility
    cap = MERGE_WINDOW_FRACTION * window_s
    if cap > 0:
        return cap
    return max(LANE_CHANGE_DURATION_S, _min_duration_for_comfort(lane_width_m))


def merge_settling_time_s(window_s: float = STEP_INTERVAL_S) -> float:
    """Simulated seconds after the lateral profile completes within one step."""
    return max(0.0, window_s - merge_duration_s(window_s=window_s))


def merge_distance_m(speed_mps: float, duration_s: float) -> float:
    """Longitudinal distance the merge consumes at the held merge speed."""
    return max(MIN_MERGE_DISTANCE_M, max(0.0, speed_mps) * duration_s)


def merge_target_speed_mps(
    current_speed_mps: float,
    *,
    max_speed_mps: float,
    min_from_rest_mps: float,
    stationary_speed_mps: float,
) -> float:
    """Speed held through the merge.

    A lane change holds speed instead of hard-braking.
    """
    floor = min(max_speed_mps, MERGE_MIN_SPEED_MPS)
    if current_speed_mps < stationary_speed_mps:
        # Ease off the line rather than sitting still mid-merge.
        return max(min_from_rest_mps * 0.7, floor)
    return max(min(max_speed_mps, current_speed_mps), floor)


def peak_lateral_accel_mps2(lane_width_m: float, duration_s: float) -> float:
    if duration_s <= 0:
        return float("inf")
    return _SMOOTHSTEP_ACCEL_PEAK * lane_width_m / (duration_s * duration_s)


def forward_travel_m(
    ego_x: float,
    ego_y: float,
    start_x: float,
    start_y: float,
    ref_yaw_deg: float,
) -> float:
    """Signed forward distance from merge start in the frozen source-lane frame."""
    yaw = math.radians(ref_yaw_deg)
    dx = ego_x - start_x
    dy = ego_y - start_y
    return dx * math.cos(yaw) + dy * math.sin(yaw)


def merge_lateral_target_m(
    src_x: float,
    src_y: float,
    right_x: float,
    right_y: float,
    lateral_offset_m: float,
    frac: float,
    side: str,
) -> tuple[float, float]:
    """Map-stable merge point: source-lane center + lateral offset along lane right."""
    sign = -1.0 if side == "left" else 1.0
    offset_m = lateral_offset_m * max(0.0, min(1.0, frac))
    return (
        src_x + sign * right_x * offset_m,
        src_y + sign * right_y * offset_m,
    )


def lateral_spacing_m(
    src_x: float,
    src_y: float,
    tgt_x: float,
    tgt_y: float,
    right_x: float,
    right_y: float,
) -> float:
    """Perpendicular distance between source and target lane centers (m)."""
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    return abs(dx * right_x + dy * right_y)


def merge_lateral_target_interpolated(
    src_x: float,
    src_y: float,
    tgt_x: float,
    tgt_y: float,
    lateral_frac: float,
) -> tuple[float, float]:
    """Interpolate along the chord from source-lane to target-lane center."""
    f = max(0.0, min(1.0, lateral_frac))
    return (
        src_x + (tgt_x - src_x) * f,
        src_y + (tgt_y - src_y) * f,
    )


def blend_yaw_deg(src_yaw: float, tgt_yaw: float, frac: float) -> float:
    """Blend headings without a 360° wrap glitch."""
    delta = _normalize_yaw_delta(tgt_yaw - src_yaw)
    blended = src_yaw + delta * max(0.0, min(1.0, frac))
    return _normalize_yaw_delta(blended)


def _normalize_yaw_delta(yaw_error_deg: float) -> float:
    while yaw_error_deg > 180.0:
        yaw_error_deg -= 360.0
    while yaw_error_deg < -180.0:
        yaw_error_deg += 360.0
    return yaw_error_deg


@dataclass
class ManeuverPlan:
    """Time-parameterized description of one committed maneuver.

    ``side`` is the lateral direction of a lane change ("left"/"right") or None
    for in-lane actions. ``is_fallback`` marks a lane change that degraded to
    lane-keeping because no adjacent driving lane exists.
    """

    action: str
    side: Optional[str]
    is_lane_change: bool
    duration_s: float
    distance_m: float
    lateral_offset_m: float
    target_speed_mps: float
    start_speed_mps: float
    source_lane_id: Optional[int] = None
    source_road_id: Optional[int] = None
    target_lane_id: Optional[int] = None
    # Frozen lane anchors at merge start (map frame). The controller advances along
    # these lanes by forward travel instead of re-projecting ego each tick.
    source_anchor_x: Optional[float] = None
    source_anchor_y: Optional[float] = None
    source_anchor_yaw: Optional[float] = None
    target_anchor_x: Optional[float] = None
    target_anchor_y: Optional[float] = None
    target_anchor_yaw: Optional[float] = None
    start_ego_x: Optional[float] = None
    start_ego_y: Optional[float] = None
    is_fallback: bool = False

    def fraction_at(self, elapsed_s: float) -> float:
        if self.duration_s <= 0:
            return 1.0
        return lateral_fraction(elapsed_s / self.duration_s)

    def is_time_complete(self, elapsed_s: float) -> bool:
        return elapsed_s >= self.duration_s


def opposite_side(side: Optional[str]) -> Optional[str]:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return None
