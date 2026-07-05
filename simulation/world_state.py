import math
import os

import carla

from simulation import step_context
from simulation.junction_planner import junction_state, lane_aware_junction_preferred_action
from simulation.maneuver_policy import compute_allowed_actions, evaluate_maneuver_policy
from simulation.pedestrian_prediction import (
    assess_nearest_pedestrian_conflict,
    choose_pedestrian_caution_action,
)

_IN_LANE_LATERAL_M = 2.5
_LEFT_LANE_RANGE = (-4.5, -1.2)
_RIGHT_LANE_RANGE = (1.2, 4.5)
_BLOCKING_MAX_LONGITUDINAL_M = float(os.getenv("BLOCKING_VEHICLE_MAX_DIST_M", "18.0"))
_SCENARIO_NPC_MAX_LONGITUDINAL_M = float(os.getenv("SCENARIO_NPC_MAX_DIST_M", "70.0"))
_DETECTION_RADIUS_M = float(os.getenv("DETECTION_RADIUS_M", "65.0"))
_MAX_ACTORS = int(os.getenv("WORLD_STATE_MAX_ACTORS", "10"))
_STATIONARY_SPEED_MPS = float(os.getenv("STATIONARY_SPEED_MPS", "0.5"))


def _is_vehicle_actor(type_id: str) -> bool:
    return type_id.startswith("vehicle.")


def _speed(velocity) -> float:
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def _ego_frame_offsets(ego_transform, location) -> tuple[float, float]:
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    rel_x = location.x - ego_transform.location.x
    rel_y = location.y - ego_transform.location.y
    longitudinal = rel_x * forward.x + rel_y * forward.y
    lateral = rel_x * right.x + rel_y * right.y
    return longitudinal, lateral


def _closing_speed(ego_velocity, actor_velocity, ego_transform, actor_transform) -> float:
    dx = actor_transform.location.x - ego_transform.location.x
    dy = actor_transform.location.y - ego_transform.location.y
    distance = math.sqrt(dx**2 + dy**2)
    if distance < 0.001:
        return 0.0
    ux = dx / distance
    uy = dy / distance
    rvx = ego_velocity.x - actor_velocity.x
    rvy = ego_velocity.y - actor_velocity.y
    return rvx * ux + rvy * uy


def _get_waypoint_info(carla_map, location) -> dict | None:
    if carla_map is None:
        return None

    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return None

    return {
        "road_id": waypoint.road_id,
        "lane_id": waypoint.lane_id,
        "lane_type": str(waypoint.lane_type),
        "is_junction": waypoint.is_junction,
        "s": round(waypoint.s, 2),
    }


def _lane_relation(ego_wp_info: dict | None, actor_wp_info: dict | None) -> dict:
    if ego_wp_info is None or actor_wp_info is None:
        return {
            "same_road": False,
            "same_lane": False,
            "lane_relation": "unknown",
        }

    same_road = ego_wp_info["road_id"] == actor_wp_info["road_id"]
    same_lane = same_road and ego_wp_info["lane_id"] == actor_wp_info["lane_id"]

    if not same_road:
        lane_relation = "different_road"
    elif same_lane:
        lane_relation = "same_lane"
    else:
        lane_relation = "neighbor_lane"

    return {
        "same_road": same_road,
        "same_lane": same_lane,
        "lane_relation": lane_relation,
    }


def _is_on_rightmost_driving_lane(waypoint) -> bool:
    """True when there is no further driving lane to the right."""
    if waypoint is None:
        return True
    right = waypoint.get_right_lane()
    return right is None or right.lane_type != carla.LaneType.Driving


def _ego_adjacent_lanes(carla_map, location) -> tuple[dict | None, bool, bool, bool]:
    if carla_map is None:
        return None, False, False, True

    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return None, False, False, True

    left = waypoint.get_left_lane()
    right = waypoint.get_right_lane()
    left_available = left is not None and left.lane_type == carla.LaneType.Driving
    right_available = right is not None and right.lane_type == carla.LaneType.Driving
    on_rightmost = _is_on_rightmost_driving_lane(waypoint)
    return (
        _get_waypoint_info(carla_map, location),
        left_available,
        right_available,
        on_rightmost,
    )


def _time_to_contact_s(distance_m: float, closing_speed_mps: float) -> float | None:
    if closing_speed_mps <= 0.1:
        return None
    return round(distance_m / closing_speed_mps, 2)


def _actor_in_ego_lane(actor: dict, lateral_m: float) -> bool:
    """True when the actor occupies ego's current driving corridor.

    Geometry (ego-frame lateral) is authoritative: CARLA ``road_id`` can differ
    along a straight segment while the NPC is still directly ahead.  After a lane
    change, a bypassed NPC in the adjacent lane has large lateral offset and is
  excluded even if map lane IDs are ambiguous.
    """
    if actor.get("same_lane"):
        return True
    return abs(lateral_m) <= _IN_LANE_LATERAL_M


def _actor_ahead_in_ego_lane(
    actor: dict,
    longitudinal_m: float,
    lateral_m: float,
    max_longitudinal_m: float,
) -> bool:
    if longitudinal_m <= 0.0 or longitudinal_m >= max_longitudinal_m:
        return False
    return _actor_in_ego_lane(actor, lateral_m)


def _build_lead_vehicle(actors: list[dict]) -> dict | None:
    """Closest in-lane vehicle ahead (scenario NPC preferred on ties)."""
    candidates: list[dict] = []
    for actor in actors:
        if not _is_vehicle_actor(actor.get("type", "")):
            continue
        ef = actor.get("ego_frame") or {}
        longitudinal = ef.get("longitudinal_m")
        lateral = ef.get("lateral_m")
        if longitudinal is None or lateral is None:
            continue
        in_lane = _actor_in_ego_lane(actor, lateral)
        if not in_lane or longitudinal <= 0:
            continue
        candidates.append(actor)

    if not candidates:
        return None

    candidates.sort(
        key=lambda actor: (
            0 if actor.get("is_scenario_npc") else 1,
            actor["ego_frame"]["longitudinal_m"],
        )
    )
    lead = candidates[0]
    ef = lead["ego_frame"]
    distance_m = ef["longitudinal_m"]
    closing = lead.get("closing_speed", 0.0)
    speed = lead.get("speed", 0.0)
    is_stationary = speed < _STATIONARY_SPEED_MPS

    return {
        "id": lead["id"],
        "type": lead["type"],
        "distance_m": round(distance_m, 2),
        "speed_mps": speed,
        "is_stationary": is_stationary,
        "in_ego_lane": _actor_in_ego_lane(lead, ef["lateral_m"]),
        "is_scenario_npc": lead.get("is_scenario_npc", False),
        "closing_speed_mps": closing,
        "time_to_contact_s": _time_to_contact_s(distance_m, closing),
        "ego_frame": ef,
    }


def _enrich_with_ego_frame(
    state: dict,
    ego_transform,
    ego_location,
    ego_speed_m_s: float,
    *,
    left_lane_available: bool,
    right_lane_available: bool,
) -> None:
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    ego_yaw_deg = ego_transform.rotation.yaw

    obstacle_ahead = False
    closest_ahead = None
    scenario_obstacle_ahead = False
    closest_scenario = None
    blocking_vehicle_ahead = False
    closest_blocking = None
    left_blocked = False
    right_blocked = False
    scenario_ids = step_context.get_scenario_npc_ids()

    for actor in state.get("nearby_actors", []):
        loc = actor["location"]
        rel_x = loc["x"] - ego_location.x
        rel_y = loc["y"] - ego_location.y
        longitudinal = rel_x * forward.x + rel_y * forward.y
        lateral = rel_x * right.x + rel_y * right.y

        velocity = actor.get("velocity") or {}
        vx = float(velocity.get("x") or 0.0)
        vy = float(velocity.get("y") or 0.0)
        longitudinal_vel = vx * forward.x + vy * forward.y
        lateral_vel = vx * right.x + vy * right.y

        actor["ego_frame"] = {
            "longitudinal_m": round(longitudinal, 2),
            "lateral_m": round(lateral, 2),
            "longitudinal_vel_mps": round(longitudinal_vel, 2),
            "lateral_vel_mps": round(lateral_vel, 2),
        }
        actor["is_scenario_npc"] = actor.get("id") in scenario_ids
        actor["is_stationary"] = actor.get("speed", 0.0) < _STATIONARY_SPEED_MPS

        if actor["is_scenario_npc"]:
            if _actor_ahead_in_ego_lane(
                actor, longitudinal, lateral, _SCENARIO_NPC_MAX_LONGITUDINAL_M
            ):
                scenario_obstacle_ahead = True
                if closest_scenario is None or longitudinal < closest_scenario:
                    closest_scenario = longitudinal
                obstacle_ahead = True
                if closest_ahead is None or longitudinal < closest_ahead:
                    closest_ahead = longitudinal

        in_ego_lane = _actor_in_ego_lane(actor, lateral)
        in_left_lane = _LEFT_LANE_RANGE[0] <= lateral <= _LEFT_LANE_RANGE[1]
        in_right_lane = _RIGHT_LANE_RANGE[0] <= lateral <= _RIGHT_LANE_RANGE[1]
        ahead_band = -5.0 < longitudinal < 45.0

        if (
            not actor["is_scenario_npc"]
            and in_ego_lane
            and 0 < longitudinal < 50.0
        ):
            obstacle_ahead = True
            if closest_ahead is None or longitudinal < closest_ahead:
                closest_ahead = longitudinal

        if (
            _is_vehicle_actor(actor.get("type", ""))
            and _actor_ahead_in_ego_lane(
                actor, longitudinal, lateral, _BLOCKING_MAX_LONGITUDINAL_M
            )
        ):
            blocking_vehicle_ahead = True
            if closest_blocking is None or longitudinal < closest_blocking:
                closest_blocking = longitudinal

        if ahead_band and in_left_lane:
            left_blocked = True
        if ahead_band and in_right_lane:
            right_blocked = True

    state["left_lane_available"] = left_lane_available
    state["right_lane_available"] = right_lane_available
    state["obstacle_ahead"] = obstacle_ahead
    state["closest_ahead_distance"] = (
        round(closest_ahead, 2) if closest_ahead is not None else None
    )
    state["scenario_obstacle_ahead"] = scenario_obstacle_ahead
    state["closest_scenario_distance"] = (
        round(closest_scenario, 2) if closest_scenario is not None else None
    )
    state["blocking_vehicle_ahead"] = blocking_vehicle_ahead
    state["closest_blocking_distance"] = (
        round(closest_blocking, 2) if closest_blocking is not None else None
    )
    state["left_lane_clear"] = left_lane_available and not left_blocked
    state["right_lane_clear"] = right_lane_available and not right_blocked

    pedestrian_scan = assess_nearest_pedestrian_conflict(
        state.get("nearby_actors", []),
        ego_yaw_deg=ego_yaw_deg,
        ego_speed_mps=ego_speed_m_s,
        follow_safe_distance_m=0.0,
        too_close_for_follow_lane=False,
    )
    if pedestrian_scan.get("pedestrian_conflict_ahead"):
        ped_dist = pedestrian_scan.get("closest_pedestrian_conflict_m")
        if ped_dist is not None:
            obstacle_ahead = True
            if closest_ahead is None or ped_dist < closest_ahead:
                closest_ahead = ped_dist
            conflict_actor_id = (pedestrian_scan.get("pedestrian_conflict") or {}).get(
                "actor_id"
            )
            if conflict_actor_id in scenario_ids:
                scenario_obstacle_ahead = True
                if closest_scenario is None or ped_dist < closest_scenario:
                    closest_scenario = ped_dist

    maneuver_obstacle = obstacle_ahead or scenario_obstacle_ahead
    maneuver_closest = closest_scenario if closest_scenario is not None else closest_ahead

    policy = evaluate_maneuver_policy(
        maneuver_obstacle,
        maneuver_closest,
        ego_speed_m_s,
        blocking_vehicle_ahead=blocking_vehicle_ahead,
        closest_blocking_m=closest_blocking,
    )
    state.update(policy)

    preferred_caution_action = None
    if pedestrian_scan.get("pedestrian_conflict_ahead") and state.get(
        "too_close_for_follow_lane"
    ):
        ped_conflict = pedestrian_scan.get("pedestrian_conflict") or {}
        ped_dist = pedestrian_scan.get("closest_pedestrian_conflict_m")
        if ped_dist is not None:
            preferred_caution_action = choose_pedestrian_caution_action(
                effective_closest_m=float(ped_dist),
                ego_speed_mps=ego_speed_m_s,
                follow_safe_distance_m=float(state.get("follow_safe_distance_m") or 0.0),
                time_to_lane_entry_s=ped_conflict.get("time_to_lane_entry_s"),
                in_lane=bool(ped_conflict.get("in_lane")),
            )
    state["preferred_caution_action"] = preferred_caution_action
    state.update(
        {
            k: pedestrian_scan[k]
            for k in (
                "pedestrian_conflict_ahead",
                "pedestrian_conflict_predicted",
                "pedestrian_conflict",
                "closest_pedestrian_conflict_m",
            )
            if k in pedestrian_scan
        }
    )

    lead_vehicle = _build_lead_vehicle(state.get("nearby_actors", []))
    state["lead_vehicle"] = lead_vehicle

    ped_conflict = state.get("pedestrian_conflict") or {}
    state["decision_hints"] = {
        "too_close_for_follow_lane": state.get("too_close_for_follow_lane"),
        "prefer_yield_or_stop": state.get("prefer_yield_or_stop"),
        "follow_safe_distance_m": state.get("follow_safe_distance_m"),
        "lead_vehicle_stationary": (
            lead_vehicle.get("is_stationary") if lead_vehicle else False
        ),
        "time_to_contact_s": (
            lead_vehicle.get("time_to_contact_s") if lead_vehicle else None
        ),
        "pedestrian_conflict_ahead": state.get("pedestrian_conflict_ahead"),
        "pedestrian_conflict_predicted": state.get("pedestrian_conflict_predicted"),
        "preferred_caution_action": state.get("preferred_caution_action"),
        "pedestrian_time_to_lane_entry_s": ped_conflict.get("time_to_lane_entry_s"),
    }

    state["allowed_actions"] = compute_allowed_actions(state, stuck=False)

    state["summary"] = {
        "num_nearby_actors": len(state.get("nearby_actors", [])),
        "detection_radius_m": state.get("detection_radius_m"),
        "closest_actor_distance": (
            state["nearby_actors"][0]["distance"]
            if state.get("nearby_actors")
            else None
        ),
        "lead_vehicle_distance_m": (
            lead_vehicle.get("distance_m") if lead_vehicle else None
        ),
        "lead_vehicle_stationary": (
            lead_vehicle.get("is_stationary") if lead_vehicle else None
        ),
    }


class WorldStateExtractor:
    def __init__(
        self,
        carla_client,
        detection_radius: float | None = None,
        max_actors: int | None = None,
    ):
        self.carla_client = carla_client
        self.detection_radius = (
            detection_radius if detection_radius is not None else _DETECTION_RADIUS_M
        )
        self.max_actors = max_actors if max_actors is not None else _MAX_ACTORS

    def get_state(self):
        world = self.carla_client.get_world()
        ego_vehicle = self.carla_client.get_ego_vehicle()

        if not world or not ego_vehicle:
            return {"error": "World or ego vehicle not initialized"}

        carla_map = world.get_map()
        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = _speed(ego_velocity)

        ego_wp_info, left_lane_available, right_lane_available, on_rightmost_lane = (
            _ego_adjacent_lanes(carla_map, ego_transform.location)
        )

        actors = world.get_actors()
        vehicles = actors.filter("vehicle.*")
        pedestrians = actors.filter("walker.*")

        nearby_actors = []
        for actor in list(vehicles) + list(pedestrians):
            if actor.id == ego_vehicle.id:
                continue

            actor_transform = actor.get_transform()
            actor_velocity = actor.get_velocity()
            distance = ego_transform.location.distance(actor_transform.location)

            if distance > self.detection_radius:
                continue

            actor_speed = _speed(actor_velocity)
            actor_wp_info = _get_waypoint_info(carla_map, actor_transform.location)
            lane_info = _lane_relation(ego_wp_info, actor_wp_info)
            closing_speed = _closing_speed(
                ego_velocity,
                actor_velocity,
                ego_transform,
                actor_transform,
            )

            nearby_actors.append({
                "id": actor.id,
                "type": actor.type_id,
                "distance": round(distance, 2),
                "speed": round(actor_speed, 2),
                "is_stationary": actor_speed < _STATIONARY_SPEED_MPS,
                "closing_speed": round(closing_speed, 2),
                "velocity": {
                    "x": round(actor_velocity.x, 2),
                    "y": round(actor_velocity.y, 2),
                    "z": round(actor_velocity.z, 2),
                },
                "same_road": lane_info["same_road"],
                "same_lane": lane_info["same_lane"],
                "lane_relation": lane_info["lane_relation"],
                "location": {
                    "x": round(actor_transform.location.x, 2),
                    "y": round(actor_transform.location.y, 2),
                    "z": round(actor_transform.location.z, 2),
                },
                "rotation": {
                    "yaw": round(actor_transform.rotation.yaw, 2),
                },
                "waypoint": actor_wp_info,
            })

        nearby_actors.sort(key=lambda actor: actor["distance"])
        nearby_actors = nearby_actors[: self.max_actors]

        on_leftmost_lane = not left_lane_available

        state = {
            "detection_radius_m": self.detection_radius,
            "on_leftmost_lane": on_leftmost_lane,
            "on_rightmost_lane": on_rightmost_lane,
            "ego_vehicle": {
                "id": ego_vehicle.id,
                "type": ego_vehicle.type_id,
                "speed": round(ego_speed, 2),
                "location": {
                    "x": round(ego_transform.location.x, 2),
                    "y": round(ego_transform.location.y, 2),
                    "z": round(ego_transform.location.z, 2),
                },
                "rotation": {
                    "yaw": round(ego_transform.rotation.yaw, 2),
                },
                "waypoint": ego_wp_info,
                "left_lane_available": left_lane_available,
                "right_lane_available": right_lane_available,
                "on_leftmost_lane": on_leftmost_lane,
                "on_rightmost_lane": on_rightmost_lane,
            },
            "nearby_actors": nearby_actors,
        }

        # Junction flags must be set before _enrich_with_ego_frame: the
        # allowed-actions computation there gates turn actions on them.
        state.update(junction_state(carla_map, ego_transform.location, ego_speed))
        if state.get("junction_ahead"):
            state["junction_preferred_action"] = lane_aware_junction_preferred_action(
                state.get("junction_options"),
                on_leftmost_lane=on_leftmost_lane,
                on_rightmost_lane=on_rightmost_lane,
            )

        _enrich_with_ego_frame(
            state,
            ego_transform,
            ego_transform.location,
            ego_speed,
            left_lane_available=left_lane_available,
            right_lane_available=right_lane_available,
        )
        return state
