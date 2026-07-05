"""Junction (intersection) handling: detection, exit options, and turn paths.

Single source of truth for what an intersection means to the agent:

- ``scan_ahead`` walks the ego lane forward and reports the upcoming junction
  (or dead end), enumerating the connector paths that leave it.
- Exits are classified ``straight`` / ``right`` / ``left`` by total heading
  change, and mapped to the discrete actions ``go_straight`` / ``turn_right``
  / ``turn_left``.
- ``preferred_junction_action`` encodes the fixed priority **forward, then
  right, then left**; when no exit exists the caller must stop before the
  junction.
- ``build_junction_plan`` freezes the chosen connector into a polyline of
  lane poses that the action executor tracks per physics tick, ending
  centered in the exit lane.

The module is import-safe without a CARLA server: only functions that touch
waypoints need CARLA objects, and ``carla`` itself is never imported at
module level so the pure helpers (classification, ordering, plan tracking)
stay unit-testable.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

from simulation import lane_controller as lc
from simulation.lane_change_controller import LanePose, lane_pose_from_waypoint
from simulation.timing_config import STEP_INTERVAL_S

# How far ahead the ego lane is scanned for a junction / dead end.
JUNCTION_DETECT_M = float(os.getenv("JUNCTION_DETECT_M", "40.0"))
# Sampling step for walking lanes and freezing turn paths.
JUNCTION_SCAN_STEP_M = float(os.getenv("JUNCTION_SCAN_STEP_M", "2.0"))
# |heading change| below this is "straight"; above JUNCTION_UTURN_MIN_DEG is a
# U-turn (never offered as an option).
JUNCTION_STRAIGHT_MAX_DEG = float(os.getenv("JUNCTION_STRAIGHT_MAX_DEG", "30.0"))
JUNCTION_UTURN_MIN_DEG = float(os.getenv("JUNCTION_UTURN_MIN_DEG", "150.0"))
# Distance a frozen turn path continues into the exit lane so the ego ends
# centered and aligned after the junction.
JUNCTION_EXIT_EXTEND_M = float(os.getenv("JUNCTION_EXIT_EXTEND_M", "12.0"))
# Upper bound on the connector length walked inside a junction.
JUNCTION_PATH_MAX_M = float(os.getenv("JUNCTION_PATH_MAX_M", "80.0"))
# Speed held while executing a junction action (approach + turn + exit).
JUNCTION_TURN_SPEED_MPS = float(os.getenv("JUNCTION_TURN_SPEED_MPS", "3.5"))
# Extra margin on top of one decision window of travel when deciding that the
# junction is "imminent" (the committed action would enter it).
JUNCTION_IMMINENT_MARGIN_M = float(os.getenv("JUNCTION_IMMINENT_MARGIN_M", "6.0"))
# Same planning-speed floor as maneuver_policy (kept independent to avoid an
# import cycle: maneuver_policy imports this module).
_MIN_PLANNING_SPEED_MPS = float(os.getenv("MIN_PLANNING_SPEED_MPS", "3.5"))
# Path tracking: lookahead along the frozen path and steering authority.
JUNCTION_LOOKAHEAD_M = float(os.getenv("JUNCTION_LOOKAHEAD_M", "4.5"))
JUNCTION_MAX_STEER = float(os.getenv("JUNCTION_MAX_STEER", "0.6"))
JUNCTION_LAT_GAIN = float(os.getenv("JUNCTION_LAT_GAIN", "0.06"))
JUNCTION_YAW_GAIN = float(os.getenv("JUNCTION_YAW_GAIN", "0.03"))
JUNCTION_MAX_YAW_ERR_DEG = float(os.getenv("JUNCTION_MAX_YAW_ERR_DEG", "60.0"))
# When |lat_err| exceeds this, yaw contribution is reduced and a lateral
# correction floor prevents corner-cutting on tight turns.
JUNCTION_OFF_PATH_TOLERANCE_M = float(
    os.getenv("JUNCTION_OFF_PATH_TOLERANCE_M", "0.5")
)
JUNCTION_YAW_WEIGHT_OFF_CENTER = float(
    os.getenv("JUNCTION_YAW_WEIGHT_OFF_CENTER", "0.12")
)
JUNCTION_MIN_STEER = float(os.getenv("JUNCTION_MIN_STEER", "0.08"))

JUNCTION_ACTIONS = ("go_straight", "turn_right", "turn_left")

ACTION_TO_DIRECTION = {
    "go_straight": "straight",
    "turn_right": "right",
    "turn_left": "left",
}
DIRECTION_TO_ACTION = {v: k for k, v in ACTION_TO_DIRECTION.items()}


def classify_turn(entry_yaw_deg: float, exit_yaw_deg: float) -> str:
    """Classify a junction connector by total heading change.

    CARLA yaw grows clockwise (left-handed, z-up), so a positive delta is a
    right turn. Returns "straight" | "right" | "left" | "u_turn".
    """
    delta = lc.normalize_yaw_error(exit_yaw_deg - entry_yaw_deg)
    if abs(delta) <= JUNCTION_STRAIGHT_MAX_DEG:
        return "straight"
    if abs(delta) >= JUNCTION_UTURN_MIN_DEG:
        return "u_turn"
    return "right" if delta > 0 else "left"


def is_single_travel_lane(
    *,
    on_leftmost_lane: bool = False,
    on_rightmost_lane: bool = False,
) -> bool:
    """True when the carriageway has only one driving lane."""
    return on_leftmost_lane and on_rightmost_lane


def filter_junction_options_by_lane(
    options: dict | None,
    *,
    on_leftmost_lane: bool = False,
    on_rightmost_lane: bool = False,
) -> dict:
    """Drop turn exits that are illegal from the ego's current lane position.

    Leftmost lane: no right turn. Rightmost lane: no left turn. Straight is
    always kept when physically available. Does not change exit priority — only
    removes lane-forbidden branches before ``preferred_junction_action``.

    On a single-lane road both edge flags are set; lane-based turn bans are
    skipped so the only physical connector (e.g. right-only) stays available.
    """
    filtered = dict(options or {})
    if is_single_travel_lane(
        on_leftmost_lane=on_leftmost_lane,
        on_rightmost_lane=on_rightmost_lane,
    ):
        return filtered
    if on_leftmost_lane:
        filtered["right"] = False
    if on_rightmost_lane:
        filtered["left"] = False
    return filtered


def preferred_junction_action(options: dict | None) -> Optional[str]:
    """Fixed priority: go forward if possible, else turn right, else left.

    Returns None when the junction offers no usable exit — the caller must
    stop before the junction.
    """
    options = options or {}
    if options.get("straight"):
        return "go_straight"
    if options.get("right"):
        return "turn_right"
    if options.get("left"):
        return "turn_left"
    return None


def lane_aware_junction_preferred_action(
    options: dict | None,
    *,
    on_leftmost_lane: bool = False,
    on_rightmost_lane: bool = False,
) -> Optional[str]:
    """``preferred_junction_action`` on lane-legal exits only."""
    return preferred_junction_action(
        filter_junction_options_by_lane(
            options,
            on_leftmost_lane=on_leftmost_lane,
            on_rightmost_lane=on_rightmost_lane,
        )
    )


def junction_imminent_distance_m(ego_speed_m_s: float) -> float:
    """Distance within which the current decision window reaches the junction."""
    planning_speed = max(ego_speed_m_s, _MIN_PLANNING_SPEED_MPS)
    return planning_speed * STEP_INTERVAL_S + JUNCTION_IMMINENT_MARGIN_M


def compute_junction_steer(
    lat_err_m: float,
    yaw_err_deg: float,
    *,
    max_steer: float = JUNCTION_MAX_STEER,
) -> float:
    """Steer along a frozen junction path; lateral-first when off the centerline."""
    yaw_err_deg = lc.normalize_yaw_error(yaw_err_deg)
    yaw_err_deg = max(
        -JUNCTION_MAX_YAW_ERR_DEG, min(JUNCTION_MAX_YAW_ERR_DEG, yaw_err_deg)
    )
    off_lat = abs(lat_err_m) > JUNCTION_OFF_PATH_TOLERANCE_M
    yaw_weight = JUNCTION_YAW_WEIGHT_OFF_CENTER if off_lat else 1.0
    steer = (-JUNCTION_LAT_GAIN * lat_err_m) + (
        JUNCTION_YAW_GAIN * yaw_err_deg * yaw_weight
    )
    if off_lat:
        sign = lc.centering_steer_sign(lat_err_m)
        floor = min(
            max_steer,
            max(JUNCTION_MIN_STEER, abs(lat_err_m) * JUNCTION_LAT_GAIN * 0.85),
        )
        if abs(steer) < floor or steer * sign < 0:
            steer = sign * floor
    return max(-max_steer, min(max_steer, steer))


def straightest_waypoint(candidates, ref_yaw_deg: float):
    """Pick the successor whose heading deviates least from ``ref_yaw_deg``.

    ``waypoint.next()`` returns one successor per branch in arbitrary order;
    inside a junction picking ``[0]`` commits to a random turn connector.
    """
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda wp: abs(
            lc.normalize_yaw_error(wp.transform.rotation.yaw - ref_yaw_deg)
        ),
    )


def _continue_branch(cur, candidates):
    """Follow one connector road through a junction without hopping branches."""
    if not candidates:
        return None
    same_road = [wp for wp in candidates if wp.road_id == cur.road_id]
    pool = same_road or candidates
    return straightest_waypoint(pool, cur.transform.rotation.yaw)


@dataclass
class JunctionScan:
    """Result of scanning the ego lane forward for the next junction."""

    kind: Optional[str] = None  # "junction" | "dead_end" | None
    distance_m: Optional[float] = None
    inside: bool = False
    # Waypoints walked along the ego lane up to (excluding) the junction.
    approach_wps: list = field(default_factory=list)
    # direction -> list of waypoints through the connector + exit extension.
    branches: dict = field(default_factory=dict)

    @property
    def options(self) -> dict:
        return {
            "straight": "straight" in self.branches,
            "right": "right" in self.branches,
            "left": "left" in self.branches,
        }


def _walk_branch(first_wp, entry_yaw_deg: float):
    """Follow a junction branch to its exit, extended into the exit lane."""
    path = [first_wp]
    cur = first_wp
    traveled = JUNCTION_SCAN_STEP_M
    while cur.is_junction and traveled < JUNCTION_PATH_MAX_M:
        nxt = _continue_branch(cur, cur.next(JUNCTION_SCAN_STEP_M))
        if nxt is None:
            return None, None
        cur = nxt
        path.append(cur)
        traveled += JUNCTION_SCAN_STEP_M
    if cur.is_junction:
        return None, None  # never left the junction within the cap
    extended = 0.0
    while extended < JUNCTION_EXIT_EXTEND_M:
        nxt = _continue_branch(cur, cur.next(JUNCTION_SCAN_STEP_M))
        if nxt is None:
            break
        cur = nxt
        path.append(cur)
        extended += JUNCTION_SCAN_STEP_M
    direction = classify_turn(entry_yaw_deg, cur.transform.rotation.yaw)
    return direction, path


def _enumerate_branches(branch_root_wp) -> dict:
    """All classified exit paths reachable from the waypoint entering a junction."""
    entry_yaw = branch_root_wp.transform.rotation.yaw
    branches: dict = {}
    for first in branch_root_wp.next(JUNCTION_SCAN_STEP_M):
        direction, path = _walk_branch(first, entry_yaw)
        if direction is None or direction == "u_turn":
            continue
        if direction in branches:
            # Keep the straighter of duplicate same-direction exits.
            existing_delta = abs(
                lc.normalize_yaw_error(
                    branches[direction][-1].transform.rotation.yaw - entry_yaw
                )
            )
            new_delta = abs(
                lc.normalize_yaw_error(path[-1].transform.rotation.yaw - entry_yaw)
            )
            if new_delta >= existing_delta:
                continue
        branches[direction] = path
    return branches


def scan_ahead(ego_wp, max_dist_m: float = JUNCTION_DETECT_M) -> JunctionScan:
    """Walk the ego lane forward and describe the next junction or dead end."""
    if ego_wp is None:
        return JunctionScan()

    if ego_wp.is_junction:
        return JunctionScan(
            kind="junction",
            distance_m=0.0,
            inside=True,
            approach_wps=[],
            branches=_enumerate_branches(ego_wp),
        )

    cur = ego_wp
    dist = 0.0
    approach = [cur]
    while dist < max_dist_m:
        candidates = cur.next(JUNCTION_SCAN_STEP_M)
        if not candidates:
            return JunctionScan(
                kind="dead_end", distance_m=round(dist, 2), approach_wps=approach
            )
        if any(wp.is_junction for wp in candidates):
            return JunctionScan(
                kind="junction",
                distance_m=round(dist, 2),
                approach_wps=approach,
                branches=_enumerate_branches(cur),
            )
        cur = straightest_waypoint(candidates, cur.transform.rotation.yaw)
        approach.append(cur)
        dist += JUNCTION_SCAN_STEP_M
    return JunctionScan()


def junction_state(carla_map, ego_location, ego_speed_m_s: float) -> dict:
    """World-state enrichment: junction flags the agent decides on."""
    import carla  # local import: keep the module importable without a simulator

    ego_wp = None
    if carla_map is not None:
        ego_wp = carla_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    scan = scan_ahead(ego_wp)
    options = scan.options
    junction_ahead = scan.kind == "junction"
    imminent = (
        scan.kind is not None
        and scan.distance_m is not None
        and scan.distance_m <= junction_imminent_distance_m(ego_speed_m_s)
    )
    return {
        "junction_ahead": junction_ahead,
        "junction_inside": scan.inside,
        "junction_kind": scan.kind,
        "junction_distance_m": scan.distance_m,
        "junction_options": options,
        "junction_preferred_action": (
            preferred_junction_action(options) if junction_ahead else None
        ),
        "junction_imminent": imminent,
        "road_end_ahead": scan.kind == "dead_end",
    }


@dataclass
class JunctionPlan:
    """Frozen path through a junction: approach + connector + exit lane."""

    action: str
    direction: str
    poses: list  # list[LanePose]
    cum_s: list  # cumulative arc length per pose
    target_speed_mps: float
    junction_distance_m: float
    exit_road_id: Optional[int] = None
    exit_lane_id: Optional[int] = None

    def nearest_index(self, x: float, y: float, start_idx: int = 0) -> int:
        """Monotonic nearest-pose search (never snaps backwards on the path)."""
        best_idx = min(start_idx, len(self.poses) - 1)
        best_d2 = None
        window_end = min(len(self.poses), best_idx + 12)
        for i in range(best_idx, window_end):
            p = self.poses[i]
            d2 = (x - p.x) ** 2 + (y - p.y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx

    def lookahead_pose(self, idx: int, lookahead_m: float = JUNCTION_LOOKAHEAD_M):
        target_s = self.cum_s[min(idx, len(self.cum_s) - 1)] + lookahead_m
        for i in range(idx, len(self.poses)):
            if self.cum_s[i] >= target_s:
                return self.poses[i]
        return self.poses[-1]

    def is_complete(self, x: float, y: float, idx: int) -> bool:
        """True once ego has passed the last frozen pose."""
        if idx < len(self.poses) - 1:
            return False
        last = self.poses[-1]
        yaw = math.radians(last.yaw_deg)
        forward = (x - last.x) * math.cos(yaw) + (y - last.y) * math.sin(yaw)
        return forward >= 0.0


def _poses_from_waypoints(waypoints) -> tuple[list, list]:
    poses = []
    for wp in waypoints:
        pose = lane_pose_from_waypoint(wp)
        if pose is not None:
            poses.append(pose)
    cum_s = [0.0]
    for prev, cur in zip(poses, poses[1:]):
        cum_s.append(cum_s[-1] + math.hypot(cur.x - prev.x, cur.y - prev.y))
    return poses, cum_s


def build_junction_plan(ego_wp, action: str) -> Optional[JunctionPlan]:
    """Freeze the full drive path for a junction action from ego's lane.

    Returns None when there is no junction ahead or the requested direction has
    no connector — the executor then falls back to a safe stop.
    """
    direction = ACTION_TO_DIRECTION.get(action)
    if direction is None:
        return None
    scan = scan_ahead(ego_wp)
    if scan.kind != "junction":
        return None
    branch = scan.branches.get(direction)
    if not branch:
        return None
    poses, cum_s = _poses_from_waypoints(scan.approach_wps + branch)
    if len(poses) < 2:
        return None
    exit_wp = branch[-1]
    return JunctionPlan(
        action=action,
        direction=direction,
        poses=poses,
        cum_s=cum_s,
        target_speed_mps=JUNCTION_TURN_SPEED_MPS,
        junction_distance_m=scan.distance_m or 0.0,
        exit_road_id=exit_wp.road_id,
        exit_lane_id=exit_wp.lane_id,
    )
