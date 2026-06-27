from simulation.lane_preference import enrich_keep_right_preference
from simulation.maneuver_policy import compute_allowed_actions, is_action_allowed


def _base_state(**overrides):
    state = {
        "path_blocked": False,
        "on_rightmost_lane": False,
        "right_lane_clear": True,
        "right_lane_available": True,
        "left_lane_clear": True,
        "left_lane_available": True,
        "maneuver_allowed": False,
        "lane_change_allowed": False,
        "lane_centering_incomplete": False,
        "lane_centering_side": None,
    }
    state.update(overrides)
    return state


def test_keep_right_prefers_change_lane_right():
    state = _base_state()
    enrich_keep_right_preference(state)
    assert state["lane_preference_allowed"] is True
    assert state["preferred_action"] == "change_lane_right"
    assert "change_lane_right" in state["allowed_actions"]


def test_on_rightmost_lane_follow_only():
    state = _base_state(on_rightmost_lane=True)
    enrich_keep_right_preference(state)
    assert state["lane_preference_allowed"] is False
    assert state["preferred_action"] == "follow_lane"
    assert "change_lane_right" not in state["allowed_actions"]
    assert "follow_lane" in state["allowed_actions"]


def test_right_lane_blocked_no_preference():
    state = _base_state(right_lane_clear=False)
    enrich_keep_right_preference(state)
    assert state["lane_preference_allowed"] is False
    assert is_action_allowed("follow_lane", state)
    assert not is_action_allowed("change_lane_right", state)


def test_path_blocked_no_keep_right_preference():
    state = _base_state(path_blocked=True)
    enrich_keep_right_preference(state)
    assert state["lane_preference_allowed"] is False
