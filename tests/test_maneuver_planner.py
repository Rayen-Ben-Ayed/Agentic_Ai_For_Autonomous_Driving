from simulation import maneuver_planner as mp
from simulation.timing_config import STEP_INTERVAL_S


def test_smoothstep_endpoints_and_clamp():
    assert mp.smoothstep(0.0) == 0.0
    assert mp.smoothstep(1.0) == 1.0
    # Clamped outside [0, 1].
    assert mp.smoothstep(-1.0) == 0.0
    assert mp.smoothstep(2.0) == 1.0
    # Symmetric midpoint.
    assert abs(mp.smoothstep(0.5) - 0.5) < 1e-9


def test_lateral_fraction_monotonic():
    samples = [mp.lateral_fraction(i / 10.0) for i in range(11)]
    assert samples[0] == 0.0
    assert samples[-1] == 1.0
    for a, b in zip(samples, samples[1:]):
        assert b >= a


def test_merge_distance_scales_with_speed_and_has_floor():
    duration = 3.5
    slow = mp.merge_distance_m(0.0, duration)
    fast = mp.merge_distance_m(10.0, duration)
    assert slow == mp.MIN_MERGE_DISTANCE_M
    assert fast > slow
    assert fast == max(mp.MIN_MERGE_DISTANCE_M, 10.0 * duration)


def test_merge_duration_is_window_fraction():
    duration = mp.merge_duration_s(3.5, STEP_INTERVAL_S)
    expected = mp.MERGE_WINDOW_FRACTION * STEP_INTERVAL_S
    assert abs(duration - expected) < 1e-9
    assert duration > 0


def test_merge_settling_time_fills_remainder_of_step():
    duration = mp.merge_duration_s(window_s=STEP_INTERVAL_S)
    settling = mp.merge_settling_time_s(STEP_INTERVAL_S)
    assert abs(duration + settling - STEP_INTERVAL_S) < 1e-9


def test_merge_target_speed_holds_speed_no_hard_brake():
    # A lane change at 10 m/s should hold ~speed, not brake to a crawl.
    ts = mp.merge_target_speed_mps(
        10.0,
        max_speed_mps=8.0,
        min_from_rest_mps=3.5,
        stationary_speed_mps=0.5,
    )
    assert ts >= 8.0 - 1e-9  # capped at max, never a hard decel target


def test_merge_target_speed_has_min_floor():
    # A near-stopped ego is lifted to the merge minimum so steering can produce
    # real lateral travel within the merge distance.
    ts = mp.merge_target_speed_mps(
        1.0,
        max_speed_mps=8.0,
        min_from_rest_mps=3.5,
        stationary_speed_mps=0.5,
    )
    assert ts == mp.MERGE_MIN_SPEED_MPS


def test_plan_fraction_progression():
    plan = mp.ManeuverPlan(
        action="change_lane_left",
        side="left",
        is_lane_change=True,
        duration_s=4.0,
        distance_m=20.0,
        lateral_offset_m=3.5,
        target_speed_mps=8.0,
        start_speed_mps=8.0,
    )
    assert plan.fraction_at(0.0) == 0.0
    assert plan.fraction_at(4.0) == 1.0
    assert plan.is_time_complete(4.0)
    assert not plan.is_time_complete(1.0)
    assert 0.0 < plan.fraction_at(2.0) < 1.0


def test_peak_lateral_accel_at_window_deadline():
    duration = mp.merge_duration_s(3.5, STEP_INTERVAL_S)
    accel = mp.peak_lateral_accel_mps2(3.5, duration)
    # Shorter 70% window raises peak accel vs a leisurely merge; still bounded.
    assert accel < 4.0


def test_opposite_side():
    assert mp.opposite_side("left") == "right"
    assert mp.opposite_side("right") == "left"
    assert mp.opposite_side(None) is None


def test_forward_travel_along_frozen_heading():
    assert abs(mp.forward_travel_m(10.0, 0.0, 0.0, 0.0, 0.0) - 10.0) < 1e-9
    assert abs(mp.forward_travel_m(0.0, 5.0, 0.0, 0.0, 90.0) - 5.0) < 1e-9
    assert abs(mp.forward_travel_m(3.0, 4.0, 0.0, 0.0, 0.0) - 3.0) < 1e-9


def test_merge_lateral_target_moves_left_of_source():
    tx, ty = mp.merge_lateral_target_m(0.0, 0.0, 1.0, 0.0, 3.5, 1.0, "left")
    assert tx < 0.0
    assert ty == 0.0


def test_merge_lateral_target_interpolated_endpoints():
    sx, sy = mp.merge_lateral_target_interpolated(0.0, 0.0, 0.0, 3.5, 0.0)
    assert sx == 0.0 and sy == 0.0
    tx, ty = mp.merge_lateral_target_interpolated(0.0, 0.0, 0.0, 3.5, 1.0)
    assert abs(tx) < 1e-9 and abs(ty - 3.5) < 1e-9
    mid_x, mid_y = mp.merge_lateral_target_interpolated(0.0, 0.0, 0.0, 4.0, 0.5)
    assert abs(mid_y - 2.0) < 1e-9


def test_lateral_spacing_m():
    spacing = mp.lateral_spacing_m(0.0, 0.0, 3.5, 0.0, 1.0, 0.0)
    assert abs(spacing - 3.5) < 1e-9


def test_blend_yaw_deg_shortest_path():
    assert abs(mp.blend_yaw_deg(350.0, 10.0, 0.5) - 0.0) < 1e-9
