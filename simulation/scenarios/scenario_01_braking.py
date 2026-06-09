import carla
import logging
from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)


class Scenario01Braking(BaseScenario):
    def __init__(self, carla_client):
        super().__init__(carla_client)

        self.cross_vehicle = None
        self.internal_step = 0
        self.llm_queried = False

        # Ego is faster now
        self.ego_throttle = 0.95

        # Scenario geometry
        self.collision_location = None
        self.cross_start_location = None
        self.cross_direction = None
        self.cross_rotation = None

        # NPC movement
        self.cross_vehicle_started = False

        # Tune these two values only if needed
        self.cross_trigger_distance = 28.0
        self.cross_speed = 10.5  # m/s, around 38 km/h

        self.cross_start_left_distance = 28.0
        self.fallback_collision_distance = 42.0

    def setup(self):
        """
        Scenario:
        Ego vehicle follows the lane and drives fast toward an intersection.
        NPC vehicle waits on the left side.
        When ego gets close, NPC crosses smoothly from left to right.
        NPC does NOT brake.
        The dangerous situation is created for the future LLM decision.
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
        logger.info("SCENARIO 01: LEFT VEHICLE INTERSECTION DANGER")
        logger.info("Ego follows lane and drives fast.")
        logger.info("NPC crosses from the left WITHOUT braking.")
        logger.info("Future LLM should detect danger and decide brake/stop/yield.")
        logger.info("Ego throttle: %.2f", self.ego_throttle)
        logger.info("NPC trigger distance: %.1f m", self.cross_trigger_distance)
        logger.info("NPC speed: %.1f m/s", self.cross_speed)
        logger.info("Conflict location: %s", self.collision_location)
        logger.info("NPC start location: %s", self.cross_start_location)
        logger.info("=================================================")

        self.world.debug.draw_string(
            self.collision_location + carla.Location(z=3.0),
            "DANGER / COLLISION POINT",
            color=carla.Color(255, 0, 0),
            life_time=30.0
        )

        self.world.debug.draw_string(
            self.cross_start_location + carla.Location(z=3.0),
            "NPC START LEFT",
            color=carla.Color(255, 128, 0),
            life_time=30.0
        )

        self.world.debug.draw_point(
            self.collision_location,
            size=0.25,
            color=carla.Color(255, 0, 0),
            life_time=30.0
        )

    def update(self, step=None):
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

        # Ego follows lane and drives fast.
        # Later, the LLM/action_executor can override this with brake/stop.
        ego.apply_control(
            carla.VehicleControl(
                throttle=self.ego_throttle,
                steer=0.0,
                brake=0.0
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

            if ego_distance < self.cross_trigger_distance:
                self.cross_vehicle_started = True

                # Release NPC once, then it drives continuously.
                self.cross_vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=0.0,
                        hand_brake=False
                    )
                )

                logger.info("NPC starts crossing from the left now.")

        else:
            # NPC moves smoothly and NEVER brakes after it starts.
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

        if step % 5 == 0:
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

        spawn_transform = carla.Transform(
            self.cross_start_location,
            self.cross_rotation
        )

        self.cross_vehicle = self.world.try_spawn_actor(
            bp,
            spawn_transform
        )

        if self.cross_vehicle is None:
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

                if current_waypoint.is_junction and distance_checked > 15.0:
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