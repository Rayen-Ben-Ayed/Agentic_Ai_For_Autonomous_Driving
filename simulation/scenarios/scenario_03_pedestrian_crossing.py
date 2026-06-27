import logging

import carla

from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)


class Scenario03PedestrianCrossing(BaseScenario):
    def __init__(self, carla_client):
        super().__init__(carla_client)

        self.pedestrian = None
        self.control_ego = False
        self.ego_throttle = 0.50
        self.internal_step = 0

        self.crossing_location = None
        self.pedestrian_start_location = None
        self.walk_direction = None

        self.pedestrian_started = False
        self.agent_trigger_distance = 34.0
        self.visual_trigger_distance = 24.0
        self.pedestrian_speed = 1.8  # m/s, brisk walking pace
        self.start_side_distance = 5.5
        self.fallback_crossing_distance = 32.0
        self._last_logged_bucket = None

    def setup(self):
        """
        Pedestrian waits at a realistic roadside/crosswalk position ahead of ego,
        then walks across from right to left.
        """
        ego = self.carla_client.get_ego_vehicle()
        if not ego:
            logger.error("Ego vehicle not found. Cannot setup scenario.")
            return

        ego_tf = ego.get_transform()
        right = ego_tf.get_right_vector()

        self.crossing_location = self._find_crossing_location(ego)
        self.pedestrian_start_location = (
            self.crossing_location + right * self.start_side_distance
        )
        self.pedestrian_start_location.z += 0.8
        self.walk_direction = carla.Vector3D(-right.x, -right.y, 0.0)

        self._spawn_pedestrian()

        logger.info("=================================================")
        logger.info("SCENARIO 03: PEDESTRIAN CROSSING")
        logger.info("Pedestrian waits at the roadside, then crosses the ego path.")
        logger.info("Ego scenario throttle: %.2f", self.ego_throttle)
        logger.info("Pedestrian trigger distance: %.1f m", self._active_trigger_distance())
        logger.info("Pedestrian speed: %.1f m/s", self.pedestrian_speed)
        logger.info("Crossing location: %s", self.crossing_location)
        logger.info("Pedestrian start location: %s", self.pedestrian_start_location)
        logger.info("=================================================")

        self.world.debug.draw_string(
            self.crossing_location + carla.Location(z=3.0),
            "PEDESTRIAN CROSSING",
            color=carla.Color(255, 0, 0),
            life_time=30.0,
        )
        self.world.debug.draw_string(
            self.pedestrian_start_location + carla.Location(z=2.5),
            "PEDESTRIAN START",
            color=carla.Color(255, 180, 0),
            life_time=30.0,
        )
        self.world.debug.draw_point(
            self.crossing_location,
            size=0.25,
            color=carla.Color(255, 0, 0),
            life_time=30.0,
        )

    def update(self, step=None, *, allow_trigger=True):
        if step is None:
            step = self.internal_step
        self.internal_step += 1

        ego = self.carla_client.get_ego_vehicle()
        if not ego:
            logger.error("Ego vehicle not found during update.")
            return
        if not self.pedestrian or not self.pedestrian.is_alive:
            logger.error("Pedestrian missing during update.")
            return

        if self.control_ego:
            ego.apply_control(
                carla.VehicleControl(throttle=self.ego_throttle, steer=0.0, brake=0.0)
            )

        ego_distance = ego.get_location().distance(self.crossing_location)
        ped_distance = self.pedestrian.get_location().distance(self.crossing_location)

        if not self.pedestrian_started:
            self._stop_pedestrian()
            if allow_trigger and ego_distance < self._active_trigger_distance():
                self.pedestrian_started = True
                logger.info(
                    "Pedestrian starts crossing now (ego->crossing=%.1fm, ped->crossing=%.1fm).",
                    ego_distance,
                    ped_distance,
                )
                self._walk_pedestrian()
        else:
            self._walk_pedestrian()

        bucket = int(ego_distance // 5)
        if bucket != self._last_logged_bucket:
            self._last_logged_bucket = bucket
            logger.info(
                "Step=%d | ego->crossing=%.1f m | ped->crossing=%.1f m | ped_started=%s",
                step,
                ego_distance,
                ped_distance,
                self.pedestrian_started,
            )

    def is_llm_needed(self, world_state):
        if self.llm_queried:
            return False
        for actor in world_state.get("nearby_actors", []):
            if actor.get("is_scenario_npc") and actor.get("distance", 999.0) < 30.0:
                self.llm_queried = True
                logger.info(
                    "Critical pedestrian detected at %.1f m — LLM should decide now.",
                    actor.get("distance"),
                )
                return True
        return False

    def _spawn_pedestrian(self):
        blueprint_library = self.world.get_blueprint_library()
        walkers = blueprint_library.filter("walker.pedestrian.*")
        if not walkers:
            logger.error("No pedestrian blueprints available.")
            return

        bp = walkers[0]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        for side_distance in (self.start_side_distance, 4.5, 6.5, 3.5):
            start = self.crossing_location - self.walk_direction * side_distance
            start.z += 0.8
            transform = carla.Transform(start, carla.Rotation(yaw=0.0))
            self.pedestrian = self.world.try_spawn_actor(bp, transform)
            if self.pedestrian is not None:
                self.pedestrian_start_location = start
                break

        if self.pedestrian is None:
            logger.error("Failed to spawn pedestrian near crossing.")
            return

        self.npc_actors.append(self.pedestrian)
        self.pedestrian.set_simulate_physics(True)
        self._stop_pedestrian()
        logger.info(
            "Pedestrian spawned successfully: id=%s type=%s",
            self.pedestrian.id,
            self.pedestrian.type_id,
        )

    def _walk_pedestrian(self):
        self.pedestrian.apply_control(
            carla.WalkerControl(
                direction=self.walk_direction,
                speed=self.pedestrian_speed,
                jump=False,
            )
        )

    def _stop_pedestrian(self):
        self.pedestrian.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(0.0, 0.0, 0.0),
                speed=0.0,
                jump=False,
            )
        )

    def _active_trigger_distance(self):
        if self.control_ego:
            return self.visual_trigger_distance
        return self.agent_trigger_distance

    def _find_crossing_location(self, ego_vehicle):
        ego_tf = ego_vehicle.get_transform()
        ego_location = ego_tf.location
        forward = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        carla_map = self.world.get_map()

        try:
            crosswalk_points = carla_map.get_crosswalks()
        except Exception:
            crosswalk_points = []

        best = None
        best_score = None
        for point in crosswalk_points:
            rel_x = point.x - ego_location.x
            rel_y = point.y - ego_location.y
            longitudinal = rel_x * forward.x + rel_y * forward.y
            lateral = rel_x * right.x + rel_y * right.y
            if not (18.0 <= longitudinal <= 45.0 and abs(lateral) <= 10.0):
                continue
            score = abs(longitudinal - self.fallback_crossing_distance) + abs(lateral) * 0.5
            if best_score is None or score < best_score:
                best = point
                best_score = score

        if best is not None:
            waypoint = carla_map.get_waypoint(
                best,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is not None:
                location = waypoint.transform.location
                location.z += 0.5
                logger.info("Using map crosswalk near ego path.")
                return location

        logger.warning(
            "No suitable crosswalk found ahead. Using marked fallback crossing %.1f m ahead.",
            self.fallback_crossing_distance,
        )
        fallback = ego_location + forward * self.fallback_crossing_distance
        fallback.z += 0.5
        return fallback
