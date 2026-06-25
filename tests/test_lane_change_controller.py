from simulation.lane_change_controller import (
    LanePose,
    compute_lane_change_steer,
    compute_post_merge_centering_steer,
    direct_lateral_fraction,
    errors_in_target_frame,
    is_centered_in_target_frame,
    merge_lead_ok,
    merge_target_xy,
    steer_toward_frozen_merge,
)


def _poses_left():
    # Lanes run along +x; right vector is +y (CARLA yaw=0 convention).
    src = LanePose(0.0, 0.0, 0.0, 0.0, 1.0)
    tgt = LanePose(0.0, 3.5, 0.0, 0.0, 1.0)
    return src, tgt


def _poses_right():
    # Merge to the right: target lane center at negative lateral offset.
    src = LanePose(0.0, 0.0, 0.0, 0.0, 1.0)
    tgt = LanePose(0.0, -3.5, 0.0, 0.0, 1.0)
    return src, tgt


def _poses():
    return _poses_left()


def test_direct_lateral_fraction_linear():
    assert direct_lateral_fraction(0.0, 2.8) == 0.0
    assert direct_lateral_fraction(1.4, 2.8) == 0.5
    assert direct_lateral_fraction(2.8, 2.8) == 1.0
    assert direct_lateral_fraction(5.0, 2.8) == 1.0


def test_merge_target_interpolates_lateral():
    src, tgt = _poses()
    x, y = merge_target_xy(src, tgt, 0.5)
    assert abs(x - 0.0) < 1e-6
    assert abs(y - 1.75) < 1e-6


def test_errors_measured_in_target_frame():
    src, tgt = _poses()
    # Ego 1m right of target lane center (positive lat in target frame).
    lat, yaw = errors_in_target_frame(0.0, 4.5, 0.0, tgt)
    assert abs(lat - 1.0) < 1e-6
    assert abs(yaw) < 1e-6


def test_lane_change_steer_pushes_toward_center():
    steer = compute_lane_change_steer(1.2, 0.0, max_steer=0.45)
    assert steer < 0  # left steer when right of lane


def test_post_merge_centering_recovers_right_overshoot():
    """debug2506_012 step 3: overshoot with small yaw must steer left, strongly."""
    steer = compute_post_merge_centering_steer(0.81, 4.2, max_steer=0.35)
    assert steer < -0.15
    assert steer * 0.81 < 0


def test_post_merge_centering_ignores_yaw_when_it_would_worsen_overshoot():
    """Mid-settle tick 112: +lat with +yaw must not command right steer."""
    steer = compute_post_merge_centering_steer(0.65, 3.3, max_steer=0.35)
    assert steer < 0


def test_post_merge_uses_stronger_authority_than_mid_merge():
    mid = compute_lane_change_steer(0.81, 4.2, max_steer=0.35)
    post = compute_post_merge_centering_steer(0.81, 4.2, max_steer=0.35)
    assert abs(post) > abs(mid)


def test_steer_toward_frozen_merge_uses_post_merge_law_at_completion():
    src, tgt = _poses_right()
    steer, lat_err, _ = steer_toward_frozen_merge(
        0.0, -2.85, 6.0, 5.3, src, tgt, 1.0
    )
    assert lat_err > 0.6
    assert steer < -0.15


def test_steer_toward_frozen_merge_right_lane():
    src, tgt = _poses_right()
    steer, lat_err, yaw_err = steer_toward_frozen_merge(
        0.0, 0.0, 0.0, 5.0, src, tgt, 1.0
    )
    assert abs(lat_err - 3.5) < 0.01
    assert abs(steer) >= 0.08


def test_steer_toward_frozen_merge_at_completion():
    src, tgt = _poses()
    # Ego on source lane, should steer toward target.
    steer, lat_err, yaw_err = steer_toward_frozen_merge(
        0.0, 0.0, 0.0, 5.0, src, tgt, 1.0
    )
    assert abs(lat_err + 3.5) < 0.01
    assert abs(steer) >= 0.08


def test_centered_when_on_target_line():
    assert is_centered_in_target_frame(0.3, 1.0)
    assert not is_centered_in_target_frame(1.0, 0.0)


def test_merge_lead_margin():
    assert merge_lead_ok(8.4, 11.31) is False
    assert merge_lead_ok(8.4, 13.0) is True
