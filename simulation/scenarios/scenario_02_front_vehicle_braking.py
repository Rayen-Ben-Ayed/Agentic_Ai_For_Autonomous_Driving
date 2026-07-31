import carla
import logging
from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)


class Scenario02FrontVehicleBraking(BaseScenario):
    def __init__(self, carla_client):
        super().__init__(carla_client)

        self.cross_vehicle = None
        self.internal_step = 0
        self.llm_queried = False
        self.control_ego = False
        self.ego_throttle = 0.72

        # Scenario geometry
        self.collision_location = None
        self.cross_start_location = None
        self.cross_direction = None
        self.cross_rotation = None

        # NPC movement
        self.cross_vehicle_started = False
        # Short, visible setup for debugging the scenario geometry first.
        self.agent_cross_trigger_distance = 34.0
        self.visual_cross_trigger_distance = 26.0
        self.cross_speed = 8.0  # m/s, around 29 km/h

        self.cross_start_left_distance = 16.0
        self.fallback_collision_distance = 38.0
        self._last_logged_bucket = None

    def setup(self):
        """
        Scenario:
        Ego vehicle drives toward a conflict point.
        NPC vehicle waits visibly on the left side.
        When ego gets close, NPC crosses smoothly from left to right.
        """

        ego = self.carla_client.get_ego_vehicle()

        if not ego:
            logger.error("Ego vehicle not found. Cannot setup scenario.")
            return

        ego_tf = ego.get_transform()
        right = ego_tf.get_right_vector()

        self.collision_location = self._find_next_junction_location(
            ego,
            fallback_distance=self.fallback_collision_distance
        )

        self.cross_start_location = (
            self.collision_location
            - right * self.cross_start_left_distance
        )
        self.cross_start_location.z += 0.5

        self.cross_direction = right

        self.cross_rotation = carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + 90.0,
            roll=0.0
        )

        self._spawn_cross_vehicle()

        logger.info("=================================================")
        logger.info("SCENARIO 02: LEFT VEHICLE INTERSECTION DANGER")
        logger.info("Ego is controlled by the LLM agent.")
        logger.info("NPC waits on the left, then crosses into the ego path.")
        logger.info("Ego scenario throttle: %.2f", self.ego_throttle)
        logger.info(
            "NPC trigger distance: %.1f m",
            self._active_trigger_distance(),
        )
        logger.info("NPC speed: %.1f m/s", self.cross_speed)
        logger.info("Conflict location: %s", self.collision_location)
        logger.info("NPC start location: %s", self.cross_start_location)
        logger.info("=================================================")

    def update(self, step=None, *, allow_trigger=True):
        if step is None:
            step = self.internal_step

        self.internal_step += 1

        ego = self.carla_client.get_ego_vehicle()

        if not ego:
            logger.error("Ego vehicle not found during update.")
            return

        if not self.cross_vehicle or not self.cross_vehicle.is_alive:
            logger.error("NPC vehicle missing during update.")
            return

        if self.control_ego:
            ego.apply_control(
                carla.VehicleControl(
                    throttle=self.ego_throttle,
                    steer=0.0,
                    brake=0.0,
                )
            )

        ego_location = ego.get_location()
        npc_location = self.cross_vehicle.get_location()

        ego_distance = ego_location.distance(self.collision_location)
        npc_distance = npc_location.distance(self.collision_location)

        # NPC waits at the left side before the critical moment.
        # This is not "braking during the crossing"; it is just initial waiting.
        if not self.cross_vehicle_started:
            self.cross_vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True
                )
            )

            if allow_trigger and ego_distance < self._active_trigger_distance():
                self.cross_vehicle_started = True

                # Release NPC once, then it drives continuously.
                self.cross_vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=0.0,
                        hand_brake=False
                    )
                )

                logger.info(
                    "NPC starts crossing now (ego->danger=%.1fm, npc->danger=%.1fm).",
                    ego_distance,
                    npc_distance,
                )
                self._drive_cross_vehicle()

        else:
            # NPC moves smoothly and NEVER brakes after it starts.
            self._drive_cross_vehicle()

        bucket = int(ego_distance // 5)
        if bucket != self._last_logged_bucket:
            self._last_logged_bucket = bucket
            logger.info(
                "Step=%d | ego->danger=%.1f m | npc->danger=%.1f m | npc_started=%s",
                step,
                ego_distance,
                npc_distance,
                self.cross_vehicle_started
            )

    def is_llm_needed(self, world_state):
        """
        Trigger the LLM once when the crossing vehicle is close.
        Later the LLM should decide: brake, stop, yield, or continue.
        """

        if self.llm_queried:
            return False

        for actor in world_state.get("nearby_actors", []):
            actor_type = actor.get("type", "")
            distance = actor.get("distance", 999.0)

            if actor_type.startswith("vehicle.") and distance < 30.0:
                self.llm_queried = True

                logger.info(
                    "Critical crossing vehicle detected at %.1f m — LLM should decide now.",
                    distance
                )

                return True

        return False

    def _spawn_cross_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()

        bp = None

        for name in [
            "vehicle.audi.tt",
            "vehicle.tesla.model3",
            "vehicle.mercedes.coupe",
            "vehicle.audi.a2"
        ]:
            try:
                bp = blueprint_library.find(name)
                break
            except Exception:
                continue

        if bp is None:
            bp = list(blueprint_library.filter("vehicle.*"))[0]

        for left_distance in (self.cross_start_left_distance, 8.0, 6.0, 4.0):
            start_location = self.collision_location - self.cross_direction * left_distance
            start_location.z += 0.8
            spawn_transform = carla.Transform(start_location, self.cross_rotation)
            self.cross_vehicle = self.world.try_spawn_actor(bp, spawn_transform)
            if self.cross_vehicle is not None:
                self.cross_start_location = start_location
                break

        if self.cross_vehicle is None:
            spawn_transform = carla.Transform(self.cross_start_location, self.cross_rotation)
            try:
                self.cross_vehicle = self.world.spawn_actor(
                    bp,
                    spawn_transform
                )
            except Exception as e:
                logger.error("Failed to spawn NPC vehicle: %s", e)
                return

        self.npc_actors.append(self.cross_vehicle)

        self.cross_vehicle.set_simulate_physics(True)
        self.cross_vehicle.set_autopilot(False)

        # Initial waiting only, before scenario danger starts.
        self.cross_vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True
            )
        )

        logger.info(
            "NPC spawned successfully: id=%s type=%s",
            self.cross_vehicle.id,
            self.cross_vehicle.type_id
        )

    def _active_trigger_distance(self):
        if self.control_ego:
            return self.visual_cross_trigger_distance
        return self.agent_cross_trigger_distance

    def _drive_cross_vehicle(self):
        self.cross_vehicle.set_target_velocity(
            carla.Vector3D(
                x=self.cross_direction.x * self.cross_speed,
                y=self.cross_direction.y * self.cross_speed,
                z=0.0
            )
        )

        self.cross_vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.80,
                steer=0.0,
                brake=0.0,
                hand_brake=False
            )
        )

    def _find_next_junction_location(self, ego_vehicle, fallback_distance):
        ego_tf = ego_vehicle.get_transform()
        ego_location = ego_tf.location
        forward = ego_tf.get_forward_vector()

        carla_map = self.world.get_map()

        waypoint = carla_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint:
            current_waypoint = waypoint
            distance_checked = 0.0

            for _ in range(80):
                next_waypoints = current_waypoint.next(1.0)

                if not next_waypoints:
                    break

                current_waypoint = next_waypoints[0]
                distance_checked += 1.0

                if current_waypoint.is_junction and distance_checked > 20.0:
                    location = current_waypoint.transform.location
                    location.z += 0.5

                    logger.info(
                        "Next junction found %.1f m ahead.",
                        distance_checked
                    )

                    return location

        logger.warning(
            "No nearby junction found. Using fallback point %.1f m ahead.",
            fallback_distance
        )

        fallback_location = ego_location + forward * fallback_distance
        fallback_location.z += 0.5

        return fallback_location

    def teardown(self):
        if self.cross_vehicle and self.cross_vehicle.is_alive:
            try:
                self.cross_vehicle.set_target_velocity(
                    carla.Vector3D(0.0, 0.0, 0.0)
                )
            except Exception as e:
                logger.warning("Could not stop NPC before teardown: %s", e)

        super().teardown()
