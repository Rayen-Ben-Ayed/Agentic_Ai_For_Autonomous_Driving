from simulation.maneuver_policy import (
    AGENT_LATENCY_S,
    STEP_INTERVAL_S,
    MIN_PLANNING_SPEED_MPS,
    STUCK_COLLISION_DELTA,
    compute_allowed_actions,
    compute_maneuver_horizon_m,
    evaluate_maneuver_policy,
    is_stuck_mode,
    allowed_actions_when_stuck,
)


def test_slow_speed_obstacle_at_49m_not_allowed():
    p = evaluate_maneuver_policy(True, 49.0, 2.3)
    assert p["maneuver_horizon_m"] < 49.0
    assert p["maneuver_allowed"] is False


def test_planning_speed_floor_at_standstill():
    horizon = compute_maneuver_horizon_m(0.0)
    expected = min(
        max(MIN_PLANNING_SPEED_MPS * (AGENT_LATENCY_S + STEP_INTERVAL_S) + 5.0, 12.0),
        40.0,
    )
    assert horizon == expected
    assert horizon > 12.0


def test_blocking_vehicle_enables_path_blocked():
    p = evaluate_maneuver_policy(
        obstacle_ahead=False,
        closest_ahead_m=None,
        ego_speed_m_s=6.0,
        blocking_vehicle_ahead=True,
        closest_blocking_m=15.0,
    )
    assert p["path_blocked"] is True
    assert p["effective_closest_distance"] == 15.0


def test_obstacle_at_6m_lateral_blocked():
    p = evaluate_maneuver_policy(True, 6.0, 10.0)
    assert p["lane_change_allowed"] is False
    assert p["prefer_yield_or_stop"] is True


def test_stuck_mode():
    assert is_stuck_mode(0.2, STUCK_COLLISION_DELTA) is True
    assert is_stuck_mode(0.2, STUCK_COLLISION_DELTA - 1) is False
    assert "stop" in allowed_actions_when_stuck()
    assert "follow_lane" not in allowed_actions_when_stuck()


def test_horizon_formula_at_speed():
    speed = 10.0
    expected = min(
        max(speed * (AGENT_LATENCY_S + STEP_INTERVAL_S) + 5.0, 12.0),
        40.0,
    )
    assert compute_maneuver_horizon_m(speed) == expected


def test_allowed_actions_excludes_follow_lane_when_too_close():
    state = {
        "path_blocked": True,
        "maneuver_allowed": True,
        "lane_change_allowed": True,
        "left_lane_clear": True,
        "right_lane_clear": True,
        "too_close_for_follow_lane": True,
    }
    allowed = compute_allowed_actions(state)
    assert "follow_lane" not in allowed
    assert "yield" in allowed
    assert "stop" in allowed
