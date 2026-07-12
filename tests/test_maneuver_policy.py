from simulation.maneuver_policy import (
    AGENT_LATENCY_S,
    STEP_INTERVAL_S,
    MIN_PLANNING_SPEED_MPS,
    STUCK_COLLISION_DELTA,
    compute_allowed_actions,
    compute_maneuver_horizon_m,
    evaluate_maneuver_policy,
    is_action_allowed,
    is_stuck_mode,
    allowed_actions_when_stuck,
)


def test_pedestrian_caution_forbids_lane_changes():
    """A crossing pedestrian (caution set) must leave only yield/stop, so the
    ego brakes instead of swerving lane-to-lane (debug0507_17)."""
    state = {
        "path_blocked": True,
        "preferred_caution_action": "stop",
        "maneuver_allowed": True,
        "lane_change_allowed": True,
        "left_lane_clear": True,
        "right_lane_clear": True,
        "left_lane_available": True,
        "right_lane_available": True,
    }
    assert compute_allowed_actions(state) == ["stop", "yield"]
    assert is_action_allowed("change_lane_right", state) is False
    assert is_action_allowed("go_straight", state) is False
    assert is_action_allowed("stop", state) is True
    assert is_action_allowed("yield", state) is True


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


def test_preference_lane_change_allowed_when_path_clear():
    state = {
        "path_blocked": False,
        "lane_preference_allowed": True,
        "preferred_action": "change_lane_right",
        "left_lane_clear": True,
        "right_lane_clear": True,
        "left_lane_available": True,
        "right_lane_available": True,
        "maneuver_allowed": False,
        "lane_change_allowed": False,
    }
    assert is_action_allowed("change_lane_right", state)
    allowed = compute_allowed_actions(state)
    assert "change_lane_right" in allowed


def test_return_requires_right_lane_clear_not_merge_heuristic():
    state = {
        "path_blocked": False,
        "lane_preference_allowed": True,
        "preferred_action": "follow_lane",
        "return_lane_side": "right",
        "left_lane_clear": True,
        "right_lane_clear": False,
        "left_lane_available": True,
        "right_lane_available": True,
        "maneuver_allowed": False,
        "lane_change_allowed": False,
        "nearby_actors": [
            {
                "type": "vehicle.npc",
                "ego_frame": {"longitudinal_m": 17.0, "lateral_m": 4.3},
            }
        ],
    }
    assert not is_action_allowed("change_lane_right", state)
    assert "change_lane_right" not in compute_allowed_actions(state)
    assert is_action_allowed("follow_lane", state)


def test_follow_lane_blocked_while_centering_incomplete():
    state = {
        "path_blocked": False,
        "lane_preference_allowed": True,
        "preferred_action": "change_lane_right",
        "lane_centering_incomplete": True,
        "lane_centering_side": "right",
        "left_lane_clear": True,
        "right_lane_clear": True,
        "left_lane_available": True,
        "right_lane_available": True,
        "maneuver_allowed": False,
        "lane_change_allowed": False,
    }
    assert not is_action_allowed("follow_lane", state)
    assert "change_lane_right" in compute_allowed_actions(state)


def _junction_state(**overrides):
    state = {
        "path_blocked": False,
        "lane_preference_allowed": False,
        "preferred_action": "follow_lane",
        "left_lane_clear": True,
        "right_lane_clear": True,
        "left_lane_available": True,
        "right_lane_available": True,
        "maneuver_allowed": False,
        "lane_change_allowed": False,
        "junction_ahead": True,
        "junction_imminent": False,
        "junction_options": {"straight": True, "right": True, "left": True},
        "junction_preferred_action": "go_straight",
    }
    state.update(overrides)
    return state


def test_junction_only_preferred_exit_allowed():
    state = _junction_state()
    assert is_action_allowed("go_straight", state)
    # Fixed order forward > right > left: lower-priority exits are rejected.
    assert not is_action_allowed("turn_right", state)
    assert not is_action_allowed("turn_left", state)


def test_junction_right_when_no_straight_exit():
    state = _junction_state(
        junction_options={"straight": False, "right": True, "left": True},
        junction_preferred_action="turn_right",
    )
    assert is_action_allowed("turn_right", state)
    assert not is_action_allowed("go_straight", state)
    assert not is_action_allowed("turn_left", state)


def test_junction_left_as_last_resort():
    state = _junction_state(
        junction_options={"straight": False, "right": False, "left": True},
        junction_preferred_action="turn_left",
    )
    assert is_action_allowed("turn_left", state)
    assert not is_action_allowed("turn_right", state)


def test_junction_leftmost_lane_forbids_right_turn():
    state = _junction_state(
        junction_options={"straight": False, "right": True, "left": True},
        junction_preferred_action="turn_left",
        on_leftmost_lane=True,
        left_lane_available=False,
    )
    assert is_action_allowed("turn_left", state)
    assert not is_action_allowed("turn_right", state)


def test_junction_rightmost_lane_forbids_left_turn():
    state = _junction_state(
        junction_options={"straight": False, "right": True, "left": True},
        junction_preferred_action="turn_right",
        on_rightmost_lane=True,
        right_lane_available=False,
    )
    assert is_action_allowed("turn_right", state)
    assert not is_action_allowed("turn_left", state)


def test_junction_actions_rejected_without_junction():
    state = _junction_state(
        junction_ahead=False,
        junction_options={"straight": False, "right": False, "left": False},
        junction_preferred_action=None,
    )
    for action in ("go_straight", "turn_right", "turn_left"):
        assert not is_action_allowed(action, state)


def test_ordinary_imminent_junction_does_not_block_follow_lane_or_lane_change():
    """A passable junction must never lock out follow_lane/lane-change.

    Regression for the debug0107_03 deadlock: an obstacle merely inside the
    (generous) follow-safety margin, combined with an imminent-but-passable
    junction, previously left only stop/yield available forever because
    follow_lane and every lateral action were blanket-vetoed. Only an actual
    dead end (no exit at all) should restrict them.
    """
    state = _junction_state(
        junction_imminent=True,
        path_blocked=True,
        maneuver_allowed=True,
        lane_change_allowed=True,
        left_lane_clear=True,
    )
    assert is_action_allowed("follow_lane", state)
    assert is_action_allowed("change_lane_left", state)
    assert is_action_allowed("change_lane_right", state)
    assert is_action_allowed("go_straight", state)


def test_junction_commit_blocks_reissuing_direction_action():
    """Round 1 is one-time: once committed, follow_lane/yield/stop take over."""
    state = _junction_state(junction_committed=True)
    assert not is_action_allowed("go_straight", state)
    assert not is_action_allowed("turn_right", state)
    assert not is_action_allowed("turn_left", state)
    assert is_action_allowed("follow_lane", state)


def test_junction_commit_does_not_block_lane_change_bailout():
    state = _junction_state(
        junction_committed=True,
        path_blocked=True,
        maneuver_allowed=True,
        lane_change_allowed=True,
        left_lane_clear=True,
    )
    assert is_action_allowed("change_lane_left", state)


def test_dead_end_leaves_only_stop_and_yield():
    """No forward/right/left exit: the agent must stop before the junction."""
    state = _junction_state(
        junction_ahead=False,
        junction_imminent=True,
        junction_options={"straight": False, "right": False, "left": False},
        junction_preferred_action=None,
        road_end_ahead=True,
    )
    assert sorted(compute_allowed_actions(state)) == ["stop", "yield"]


def test_no_exit_junction_leaves_only_stop_and_yield():
    state = _junction_state(
        junction_imminent=True,
        junction_options={"straight": False, "right": False, "left": False},
        junction_preferred_action=None,
    )
    assert sorted(compute_allowed_actions(state)) == ["stop", "yield"]


def test_single_lane_right_only_junction_allows_turn_right():
    """Reproduces debug0507_20 steps 11+ policy deadlock."""
    state = _junction_state(
        junction_imminent=True,
        junction_options={"straight": False, "right": True, "left": False},
        junction_preferred_action="turn_right",
        on_leftmost_lane=True,
        on_rightmost_lane=True,
        left_lane_available=False,
        right_lane_available=False,
        left_lane_clear=False,
        right_lane_clear=False,
        lane_centering_incomplete=True,
        lane_centering_side="right",
    )
    allowed = compute_allowed_actions(state)
    assert allowed == ["turn_right"]


def test_junction_round1_clear_path_only_commits_preferred_exit():
    """debug0507_21: LLM must not yield/stop when the path is clear at round 1."""
    state = _junction_state(
        junction_imminent=True,
        junction_options={"straight": False, "right": True, "left": False},
        junction_preferred_action="turn_right",
        on_rightmost_lane=True,
        right_lane_available=False,
    )
    assert compute_allowed_actions(state) == ["turn_right"]
    assert not is_action_allowed("yield", state)
    assert not is_action_allowed("stop", state)
    assert not is_action_allowed("follow_lane", state)


def test_junction_action_rejected_when_hazard_too_close():
    state = _junction_state(
        path_blocked=True,
        too_close_for_follow_lane=True,
    )
    assert not is_action_allowed("go_straight", state)
    assert is_action_allowed("yield", state)


def test_debug0107_03_scenario_does_not_deadlock():
    """Reproduces the exact debug0107_03 run: scenario 1, stationary NPC ~34m
    ahead. With this deployment's MANEUVER_SAFETY_MARGIN_M=24.0 override, the
    follow-safety distance reaches 41.5m, so too_close_for_follow_lane trips
    well before the NPC is a real hazard — while a (passable) junction sits
    right at the spawn point. The agent must still be able to move — via
    change_lane_left — instead of only stop/yield forever, which is exactly
    what happened before this fix (allowed=stop,yield for steps 2-7, parked
    at x=-50.9 for the rest of the run).
    """
    state = {
        # From the run log: path_blocked=True, maneuver_ok=True, lane_chg_ok=True,
        # but too_close_for_follow_lane=True (33.67m < the 41.5m safe-follow
        # distance under the deployed MANEUVER_SAFETY_MARGIN_M=24.0).
        "path_blocked": True,
        "maneuver_allowed": True,
        "lane_change_allowed": True,
        "too_close_for_follow_lane": True,
        "left_lane_clear": True,
        "right_lane_clear": False,
        "left_lane_available": True,
        "right_lane_available": False,
        "junction_ahead": True,
        "junction_imminent": True,
        "junction_committed": False,
        "junction_options": {"straight": True, "right": False, "left": False},
        "junction_preferred_action": "go_straight",
        "preferred_action": "follow_lane",
        "lane_preference_allowed": False,
    }
    allowed = compute_allowed_actions(state)
    assert allowed != ["stop", "yield"], "agent must not deadlock on stop/yield"
    assert "change_lane_left" in allowed
