import math
import os

from simulation import step_context
from simulation.maneuver_policy import evaluate_maneuver_policy

_IN_LANE_LATERAL_M = 2.5
_LEFT_LANE_RANGE = (-4.5, -1.2)
_RIGHT_LANE_RANGE = (1.2, 4.5)
_BLOCKING_MAX_LONGITUDINAL_M = float(os.getenv("BLOCKING_VEHICLE_MAX_DIST_M", "18.0"))
_BLOCKING_MAX_LATERAL_M = float(os.getenv("BLOCKING_VEHICLE_MAX_LATERAL_M", "12.0"))
_SCENARIO_NPC_MAX_LONGITUDINAL_M = float(os.getenv("SCENARIO_NPC_MAX_DIST_M", "70.0"))
_SCENARIO_NPC_MAX_LATERAL_M = float(os.getenv("SCENARIO_NPC_MAX_LATERAL_M", "6.0"))


def _is_vehicle_actor(type_id: str) -> bool:
    return type_id.startswith("vehicle.")


def _enrich_with_ego_frame(state: dict, ego_transform, ego_location, ego_speed_m_s: float) -> None:
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()

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

        actor["ego_frame"] = {
            "longitudinal_m": round(longitudinal, 2),
            "lateral_m": round(lateral, 2),
        }
        actor["is_scenario_npc"] = actor.get("id") in scenario_ids

        if actor["is_scenario_npc"]:
            in_lane_ahead = (
                0.0 < longitudinal < _SCENARIO_NPC_MAX_LONGITUDINAL_M
                and abs(lateral) <= _SCENARIO_NPC_MAX_LATERAL_M
            )
            if in_lane_ahead:
                scenario_obstacle_ahead = True
                if closest_scenario is None or longitudinal < closest_scenario:
                    closest_scenario = longitudinal
                if abs(lateral) <= _IN_LANE_LATERAL_M:
                    obstacle_ahead = True
                    if closest_ahead is None or longitudinal < closest_ahead:
                        closest_ahead = longitudinal

        in_ego_lane = abs(lateral) <= _IN_LANE_LATERAL_M
        in_left_lane = _LEFT_LANE_RANGE[0] <= lateral <= _LEFT_LANE_RANGE[1]
        in_right_lane = _RIGHT_LANE_RANGE[0] <= lateral <= _RIGHT_LANE_RANGE[1]
        ahead_band = -5.0 < longitudinal < 45.0

        if in_ego_lane and 0 < longitudinal < 50.0:
            obstacle_ahead = True
            if closest_ahead is None or longitudinal < closest_ahead:
                closest_ahead = longitudinal

        if (
            _is_vehicle_actor(actor.get("type", ""))
            and 0 < longitudinal < _BLOCKING_MAX_LONGITUDINAL_M
            and abs(lateral) <= _BLOCKING_MAX_LATERAL_M
        ):
            blocking_vehicle_ahead = True
            if closest_blocking is None or longitudinal < closest_blocking:
                closest_blocking = longitudinal

        if ahead_band and in_left_lane:
            left_blocked = True
        if ahead_band and in_right_lane:
            right_blocked = True

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
    state["left_lane_clear"] = not left_blocked
    state["right_lane_clear"] = not right_blocked

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


class WorldStateExtractor:
    def __init__(self, carla_client):
        self.carla_client = carla_client

    def get_state(self):
        world = self.carla_client.get_world()
        ego_vehicle = self.carla_client.get_ego_vehicle()

        if not world or not ego_vehicle:
            return {"error": "World or ego vehicle not initialized"}

        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)

        actors = world.get_actors()
        vehicles = actors.filter("vehicle.*")
        pedestrians = actors.filter("walker.*")

        nearby_actors = []
        for actor in list(vehicles) + list(pedestrians):
            if actor.id == ego_vehicle.id:
                continue

            actor_transform = actor.get_transform()
            distance = ego_transform.location.distance(actor_transform.location)

            if distance < 50.0:
                actor_velocity = actor.get_velocity()
                actor_speed = math.sqrt(
                    actor_velocity.x**2 + actor_velocity.y**2 + actor_velocity.z**2
                )

                nearby_actors.append({
                    "id": actor.id,
                    "type": actor.type_id,
                    "distance": round(distance, 2),
                    "speed": round(actor_speed, 2),
                    "location": {
                        "x": round(actor_transform.location.x, 2),
                        "y": round(actor_transform.location.y, 2),
                    },
                })

        state = {
            "ego_vehicle": {
                "speed": round(ego_speed, 2),
                "location": {
                    "x": round(ego_transform.location.x, 2),
                    "y": round(ego_transform.location.y, 2),
                },
                "rotation": {
                    "yaw": round(ego_transform.rotation.yaw, 2),
                },
            },
            "nearby_actors": nearby_actors,
        }

        _enrich_with_ego_frame(state, ego_transform, ego_transform.location, ego_speed)
        return state
