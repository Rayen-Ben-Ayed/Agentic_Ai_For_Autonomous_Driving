from simulation.lane_controller import (
    compute_centering_steer,
    compute_steer,
    lateral_error_m,
    lateral_weight_for_yaw,
    normalize_yaw_error,
    speed_scaled_max_steer,
)


def test_normalize_yaw():
    assert normalize_yaw_error(190) == -170


def test_lateral_error_sign():
    # Ego 1m to the right of target -> positive lateral -> left steer (negative)
    lat = lateral_error_m(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert lat == 1.0
    steer = compute_steer(lat, 0.0, lat_gain=0.05, yaw_gain=0.02, max_steer=0.5)
    assert steer < 0


def test_yaw_alignment_steer():
    steer = compute_steer(0.0, 10.0, lat_gain=0.05, yaw_gain=0.02, max_steer=0.5)
    assert steer > 0


def test_steer_clamped():
    steer = compute_steer(20.0, 0.0, lat_gain=1.0, yaw_gain=0.0, max_steer=0.3)
    assert steer == -0.3


def test_lateral_weight_fades_with_yaw():
    assert lateral_weight_for_yaw(50.0) == 0.0
    assert lateral_weight_for_yaw(10.0) == 1.0


def test_speed_scaled_steer_cap():
    assert speed_scaled_max_steer(0.0, lane_change=False) <= 0.18
    assert speed_scaled_max_steer(10.0, lane_change=False) <= 0.18


def test_lane_change_low_speed_has_authority():
    # At ~2.8 m/s the old cap was ~0.085 (too weak to cover a lane width).
    # The raised lane-change floor/scale should give meaningfully more authority.
    cap = speed_scaled_max_steer(2.81, lane_change=True)
    assert cap > 0.18
    # Lane-change authority should exceed lane-keeping at the same speed.
    assert cap > speed_scaled_max_steer(2.81, lane_change=False)


def test_centering_steer_avoids_yaw_lat_cancellation():
    # Run-7 case: lat=-0.75m, yaw=-1.2° — plain P+D terms nearly cancel.
    plain = compute_steer(
        -0.75,
        -1.2,
        lat_gain=0.045,
        yaw_gain=0.028,
        max_steer=0.5,
    )
    assert abs(plain) < 0.01
    centered = compute_centering_steer(
        -0.75,
        -1.2,
        lat_gain=0.045,
        yaw_gain=0.028,
        max_steer=0.5,
        lat_tolerance_m=0.7,
        min_steer=0.05,
    )
    assert centered > 0.04


def test_centering_steer_scales_with_large_error():
    steer = compute_centering_steer(
        -2.0,
        -5.0,
        lat_gain=0.045,
        yaw_gain=0.028,
        max_steer=0.5,
        lat_tolerance_m=0.7,
        min_steer=0.05,
    )
    assert steer > 0.055


def test_centering_steer_sign_when_right_of_lane():
    steer = compute_centering_steer(
        0.9,
        0.0,
        lat_gain=0.045,
        yaw_gain=0.028,
        max_steer=0.5,
        lat_tolerance_m=0.7,
        min_steer=0.05,
    )
    assert steer < -0.04


def test_curve_off_center_plain_steer_steers_wrong_way():
    """debug1007_01 step-16: +3.4m lat, +15° yaw — plain follow steers right."""
    plain = compute_steer(
        3.38,
        15.0,
        lat_gain=0.045,
        yaw_gain=0.016,
        max_steer=0.097,
    )
    assert plain > 0.05

    centered = compute_centering_steer(
        3.38,
        15.0,
        lat_gain=0.045,
        yaw_gain=0.028,
        max_steer=0.28,
        lat_tolerance_m=0.7,
        min_steer=0.05,
    )
    assert centered < -0.05
    assert centered * plain < 0  # opposite sign — recovery vs drift
