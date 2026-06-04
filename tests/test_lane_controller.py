from simulation.lane_controller import (
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
