import math
from types import SimpleNamespace

from simulation.junction_planner import (
    JUNCTION_IMMINENT_MARGIN_M,
    JUNCTION_LOOKAHEAD_M,
    JunctionPlan,
    classify_turn,
    compute_junction_steer,
    filter_junction_options_by_lane,
    junction_imminent_distance_m,
    lane_aware_junction_preferred_action,
    preferred_junction_action,
    straightest_waypoint,
)
from simulation.lane_change_controller import LanePose
from simulation.timing_config import STEP_INTERVAL_S


def test_classify_turn_straight():
    assert classify_turn(0.0, 0.0) == "straight"
    assert classify_turn(10.0, 35.0) == "straight"


def test_classify_turn_right_positive_delta():
    # CARLA yaw grows clockwise: exit right of entry has positive delta.
    assert classify_turn(0.0, 90.0) == "right"
    assert classify_turn(45.0, 130.0) == "right"


def test_classify_turn_left_negative_delta():
    assert classify_turn(0.0, -90.0) == "left"


def test_classify_turn_u_turn():
    assert classify_turn(0.0, 179.0) == "u_turn"
    assert classify_turn(0.0, -170.0) == "u_turn"


def test_classify_turn_yaw_wraparound():
    # 170° -> -170° is a 20° right bend across the ±180 seam, not a U-turn.
    assert classify_turn(170.0, -170.0) == "straight"
    assert classify_turn(-170.0, -80.0) == "right"


def test_preferred_action_order_forward_first():
    all_open = {"straight": True, "right": True, "left": True}
    assert preferred_junction_action(all_open) == "go_straight"


def test_preferred_action_right_before_left():
    assert (
        preferred_junction_action({"straight": False, "right": True, "left": True})
        == "turn_right"
    )


def test_preferred_action_left_last():
    assert (
        preferred_junction_action({"straight": False, "right": False, "left": True})
        == "turn_left"
    )


def test_preferred_action_none_when_no_exit():
    assert preferred_junction_action({"straight": False, "right": False}) is None
    assert preferred_junction_action({}) is None
    assert preferred_junction_action(None) is None


def test_filter_junction_options_leftmost_blocks_right():
    options = {"straight": True, "right": True, "left": True}
    filtered = filter_junction_options_by_lane(options, on_leftmost_lane=True)
    assert filtered == {"straight": True, "right": False, "left": True}


def test_filter_junction_options_rightmost_blocks_left():
    options = {"straight": False, "right": True, "left": True}
    filtered = filter_junction_options_by_lane(options, on_rightmost_lane=True)
    assert filtered == {"straight": False, "right": True, "left": False}


def test_lane_aware_preferred_skips_forbidden_right_on_leftmost():
    options = {"straight": False, "right": True, "left": True}
    assert (
        lane_aware_junction_preferred_action(options, on_leftmost_lane=True)
        == "turn_left"
    )


def test_lane_aware_preferred_skips_forbidden_left_on_rightmost():
    options = {"straight": False, "right": True, "left": True}
    assert (
        lane_aware_junction_preferred_action(options, on_rightmost_lane=True)
        == "turn_right"
    )


def test_single_lane_right_only_junction_allows_turn_right():
    """debug0507_20: one-lane approach to a right-only exit must not deadlock."""
    options = {"straight": False, "right": True, "left": False}
    filtered = filter_junction_options_by_lane(
        options, on_leftmost_lane=True, on_rightmost_lane=True
    )
    assert filtered == options
    assert (
        lane_aware_junction_preferred_action(
            options, on_leftmost_lane=True, on_rightmost_lane=True
        )
        == "turn_right"
    )


def test_imminent_distance_covers_one_decision_window():
    speed = 5.0
    assert junction_imminent_distance_m(speed) == (
        speed * STEP_INTERVAL_S + JUNCTION_IMMINENT_MARGIN_M
    )


def test_imminent_distance_uses_planning_floor_at_rest():
    assert junction_imminent_distance_m(0.0) > JUNCTION_IMMINENT_MARGIN_M


def test_junction_steer_signs():
    # Target heading to the right (positive yaw error) -> steer right (positive).
    assert compute_junction_steer(0.0, 30.0) > 0
    # Ego right of the path (positive lateral error) -> steer left (negative).
    assert compute_junction_steer(1.5, 0.0) < 0
    # Clamped to max authority.
    assert abs(compute_junction_steer(-10.0, 60.0)) <= 0.6


def test_junction_steer_lateral_first_when_off_path():
    # Far right of path with yaw still demanding a right turn — must correct left.
    assert compute_junction_steer(4.2, 24.0) < 0
    # On-path: yaw still dominates for turn tracking.
    assert compute_junction_steer(0.2, 30.0) > 0


def _fake_wp(yaw_deg):
    return SimpleNamespace(
        transform=SimpleNamespace(rotation=SimpleNamespace(yaw=yaw_deg))
    )


def test_straightest_waypoint_prefers_smallest_heading_change():
    left = _fake_wp(-85.0)
    straight = _fake_wp(3.0)
    right = _fake_wp(88.0)
    assert straightest_waypoint([left, straight, right], 0.0) is straight
    assert straightest_waypoint([], 0.0) is None


def _straight_plan(length_m=20.0, step=2.0):
    poses = []
    cum_s = []
    x = 0.0
    while x <= length_m:
        poses.append(LanePose(x, 0.0, 0.0, 0.0, 1.0))
        cum_s.append(x)
        x += step
    return JunctionPlan(
        action="go_straight",
        direction="straight",
        poses=poses,
        cum_s=cum_s,
        target_speed_mps=3.5,
        junction_distance_m=10.0,
    )


def test_plan_nearest_index_is_monotonic():
    plan = _straight_plan()
    idx = plan.nearest_index(5.0, 0.2, 0)
    assert plan.poses[idx].x == 4.0
    # Never snaps backwards even if a point behind is closer.
    assert plan.nearest_index(0.0, 0.0, idx) == idx


def test_plan_lookahead_pose_is_ahead_by_arc_length():
    plan = _straight_plan()
    idx = plan.nearest_index(4.0, 0.0, 0)
    ref = plan.lookahead_pose(idx)
    assert ref.x >= 4.0 + JUNCTION_LOOKAHEAD_M


def test_plan_completion_only_past_last_pose():
    plan = _straight_plan()
    last_idx = len(plan.poses) - 1
    assert not plan.is_complete(10.0, 0.0, 3)
    assert not plan.is_complete(plan.poses[-1].x - 1.0, 0.0, last_idx)
    assert plan.is_complete(plan.poses[-1].x + 0.5, 0.0, last_idx)


def test_plan_tracks_a_right_turn_geometry():
    """Quarter-circle to the right: tracking pose yaw grows toward +90."""
    radius = 8.0
    poses = []
    cum_s = [0.0]
    n = 12
    for i in range(n + 1):
        theta = (math.pi / 2) * i / n
        x = radius * math.sin(theta)
        y = radius * (1.0 - math.cos(theta))
        yaw = math.degrees(theta)
        poses.append(
            LanePose(
                x,
                y,
                yaw,
                -math.sin(math.radians(yaw)),
                math.cos(math.radians(yaw)),
            )
        )
        if i:
            prev = poses[i - 1]
            cum_s.append(cum_s[-1] + math.hypot(x - prev.x, y - prev.y))
    plan = JunctionPlan(
        action="turn_right",
        direction="right",
        poses=poses,
        cum_s=cum_s,
        target_speed_mps=3.5,
        junction_distance_m=0.0,
    )
    idx = plan.nearest_index(poses[3].x, poses[3].y, 0)
    ref = plan.lookahead_pose(idx)
    # The lookahead reference heads further into the turn than the ego pose.
    assert ref.yaw_deg > poses[idx].yaw_deg
    # A yaw error toward the turn produces right steer.
    steer = compute_junction_steer(0.0, ref.yaw_deg - poses[idx].yaw_deg)
    assert steer > 0
