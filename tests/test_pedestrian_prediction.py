from simulation.pedestrian_prediction import (
    assess_nearest_pedestrian_conflict,
    choose_pedestrian_caution_action,
    evaluate_pedestrian_conflict,
)


def test_crossing_pedestrian_outside_lane_is_predicted():
    """Regression for debug0507_02ped step-2 geometry (lat=3.68, moving left)."""
    result = evaluate_pedestrian_conflict(
        longitudinal_m=12.89,
        lateral_m=3.68,
        lateral_vel_mps=-1.8,
        longitudinal_vel_mps=0.0,
        speed_mps=1.8,
        yaw_deg=270.0,
        ego_yaw_deg=0.2,
        prediction_horizon_s=4.0,
    )
    assert result["conflict"] is True
    assert result["predicted"] is True
    assert result["time_to_lane_entry_s"] is not None
    assert result["time_to_lane_entry_s"] < 1.0


def test_road_crossing_flagged_before_ego_lane_entry():
    """Regression for debug0507_15 step-2: a walker sweeping across the road far
    outside the ego lane (won't reach the lane within the horizon) must still be
    flagged as a conflict — react to the crossing, not just lane entry."""
    result = evaluate_pedestrian_conflict(
        longitudinal_m=14.88,
        lateral_m=14.22,
        lateral_vel_mps=-1.8,
        longitudinal_vel_mps=0.0,
        speed_mps=1.8,
        yaw_deg=270.0,
        ego_yaw_deg=-25.8,
    )
    assert result["conflict"] is True
    assert result["crossing_road"] is True
    assert result["in_lane"] is False


def test_far_road_crossing_beyond_range_not_flagged():
    """A crossing walker too far ahead is out of reach and not yet a conflict."""
    result = evaluate_pedestrian_conflict(
        longitudinal_m=60.0,
        lateral_m=12.0,
        lateral_vel_mps=-1.8,
        longitudinal_vel_mps=0.0,
        speed_mps=1.8,
        yaw_deg=270.0,
        ego_yaw_deg=0.0,
    )
    assert result["conflict"] is False
    assert result["crossing_road"] is False


def test_parallel_pedestrian_on_sidewalk_not_predicted():
    result = evaluate_pedestrian_conflict(
        longitudinal_m=20.0,
        lateral_m=6.0,
        lateral_vel_mps=0.0,
        longitudinal_vel_mps=1.5,
        speed_mps=1.5,
        yaw_deg=0.0,
        ego_yaw_deg=0.0,
        prediction_horizon_s=4.0,
    )
    assert result["conflict"] is False
    assert result["predicted"] is False


def test_pedestrian_already_in_lane_is_conflict():
    result = evaluate_pedestrian_conflict(
        longitudinal_m=10.0,
        lateral_m=1.2,
        lateral_vel_mps=-1.0,
        longitudinal_vel_mps=0.0,
        speed_mps=1.0,
        yaw_deg=270.0,
        ego_yaw_deg=0.0,
        prediction_horizon_s=4.0,
    )
    assert result["conflict"] is True
    assert result["in_lane"] is True
    assert result["predicted"] is False


def test_choose_stop_when_ped_enters_before_ego_arrives():
    action = choose_pedestrian_caution_action(
        effective_closest_m=12.0,
        ego_speed_mps=5.0,
        follow_safe_distance_m=25.0,
        time_to_lane_entry_s=0.7,
        in_lane=False,
    )
    assert action == "stop"


def test_choose_yield_when_farther_but_follow_forbidden():
    action = choose_pedestrian_caution_action(
        effective_closest_m=18.0,
        ego_speed_mps=3.0,
        follow_safe_distance_m=25.0,
        time_to_lane_entry_s=8.0,
        in_lane=False,
    )
    assert action == "yield"


def test_assess_nearest_conflict_on_actor_list():
    actors = [
        {
            "id": 30,
            "type": "walker.pedestrian.0037",
            "speed": 1.8,
            "rotation": {"yaw": 270.0},
            "ego_frame": {
                "longitudinal_m": 12.89,
                "lateral_m": 3.68,
                "lateral_vel_mps": -1.8,
                "longitudinal_vel_mps": 0.0,
            },
        }
    ]
    scan = assess_nearest_pedestrian_conflict(
        actors,
        ego_yaw_deg=0.2,
        ego_speed_mps=5.1,
    )
    assert scan["pedestrian_conflict_ahead"] is True
    assert scan["pedestrian_conflict_predicted"] is True
    assert scan["closest_pedestrian_conflict_m"] == 12.89
    assert actors[0]["pedestrian_assessment"]["predicted"] is True
