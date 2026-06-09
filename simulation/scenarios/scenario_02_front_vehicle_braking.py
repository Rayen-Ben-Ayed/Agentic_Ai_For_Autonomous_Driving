import carla
import logging
from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)


class Scenario02FrontVehicleBraking(BaseScenario):
    def __init__(self, carla_client):
        super().__init__(carla_client)

        self.front_vehicle = None
        self.internal_step = 0
        self.llm_queried = False

        # Important for LLM handover
        self.agent_has_control = False

        # Ego car: not too slow, otherwise collision takes too long
        self.ego_throttle = 0.70

        # Front vehicle starts closer, so the event happens earlier
        self.front_vehicle_spawn_distance = 32.0

        # Front vehicle moves slowly first
        self.front_vehicle_initial_speed = 4.0  # m/s, around 14 km/h

        # Front vehicle stops early
        self.braking_started = False
        self.brake_trigger_distance = 30.0

        # LLM trigger later, close to the dangerous moment
        # This allows you to see collision if brake is too late.
        self.llm_trigger_distance = 10.0

        self.collision_happened = False

    def setup(self):
        """
        Scenario 02:
        Ego follows the lane.
        Front vehicle is ahead.
        Front vehicle stops early.
        Ego approaches and a rear-end collision happens earlier.
        The LLM decision point is prepared close to the danger moment.
        """

        ego = self.carla_client.get_ego_vehicle()

        if not ego:
            logger.error("Ego vehicle not found. Cannot setup scenario.")
            return

        ego_tf = ego.get_transform()
        forward = ego_tf.get_forward_vector()

        spawn_location = ego_tf.location + forward * self.front_vehicle_spawn_distance
        spawn_location.z += 0.5

        spawn_rotation = ego_tf.rotation

        self._spawn_front_vehicle(carla.Transform(spawn_location, spawn_rotation))

        logger.info("=================================================")
        logger.info("SCENARIO 02: EARLY FRONT CAR STOP + VISIBLE COLLISION")
        logger.info("Ego follows lane and approaches the front vehicle.")
        logger.info("Front vehicle stops very early.")
        logger.info("Collision should happen earlier and be visible.")
        logger.info("Ego throttle: %.2f", self.ego_throttle)
        logger.info("Front spawn distance: %.1f m", self.front_vehicle_spawn_distance)
        logger.info("Front brake trigger distance: %.1f m", self.brake_trigger_distance)
        logger.info("LLM trigger distance: %.1f m", self.llm_trigger_distance)
        logger.info("=================================================")

        self.world.debug.draw_string(
            spawn_location + carla.Location(z=3.0),
            "FRONT CAR - WILL STOP EARLY",
            color=carla.Color(255, 128, 0),
            life_time=120.0
        )

    def update(self, step=None):
        if step is None:
            step = self.internal_step

        self.internal_step += 1

        ego = self.carla_client.get_ego_vehicle()

        if not ego:
            logger.error("Ego vehicle not found during update.")
            return

        if not self.front_vehicle or not self.front_vehicle.is_alive:
            logger.error("Front vehicle missing during update.")
            return

        ego_tf = ego.get_transform()
        ego_forward = ego_tf.get_forward_vector()

        ego_location = ego.get_location()
        front_location = self.front_vehicle.get_location()

        distance = ego_location.distance(front_location)

        # Detect collision visually/logically
        if distance < 3.0 and not self.collision_happened:
            self.collision_happened = True
            logger.info("=================================================")
            logger.info("COLLISION / VERY CLOSE CONTACT HAPPENED.")
            logger.info("Distance ego-front: %.2f m", distance)
            logger.info("Keep simulation running to observe the result.")
            logger.info("=================================================")

        # Ego drives forward until LLM/agent takes control.
        # If the LLM says brake, main.py should call give_control_to_agent()
        # and action_executor.execute_action('brake').
        if not self.agent_has_control:
            ego.apply_control(
                carla.VehicleControl(
                    throttle=self.ego_throttle,
                    steer=0.0,
                    brake=0.0
                )
            )

        # Front vehicle phase 1: moves slowly
        if not self.braking_started:
            self.front_vehicle.set_target_velocity(
                carla.Vector3D(
                    x=ego_forward.x * self.front_vehicle_initial_speed,
                    y=ego_forward.y * self.front_vehicle_initial_speed,
                    z=0.0
                )
            )

            self.front_vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.18,
                    steer=0.0,
                    brake=0.0,
                    hand_brake=False
                )
            )

            # Because brake_trigger_distance is 30 m and spawn is 32 m,
            # the front car stops almost immediately.
            if distance < self.brake_trigger_distance:
                self.braking_started = True

                logger.info("=================================================")
                logger.info("FRONT VEHICLE STOPS EARLY NOW.")
                logger.info("Distance ego-front: %.1f m", distance)
                logger.info("=================================================")

        else:
            # Front vehicle remains stopped.
            self.front_vehicle.set_target_velocity(
                carla.Vector3D(0.0, 0.0, 0.0)
            )

            self.front_vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    hand_brake=True
                )
            )

        if step % 10 == 0:
            logger.info(
                "Step=%d | distance ego-front=%.1f m | front_braking=%s | agent_has_control=%s | collision=%s",
                step,
                distance,
                self.braking_started,
                self.agent_has_control,
                self.collision_happened
            )

    def give_control_to_agent(self):
        """
        Called by main.py when LLM decision is taken.
        After this, scenario stops forcing ego throttle.
        """
        self.agent_has_control = True
        logger.info("Scenario handed ego control to the agent/LLM.")

    def is_llm_needed(self, world_state):
        """
        Trigger LLM close to the danger moment.
        For this test, the trigger is intentionally later so you can see
        whether braking is too late or whether collision still happens.
        """

        if self.llm_queried:
            return False

        for actor in world_state.get("nearby_actors", []):
            actor_type = actor.get("type", "")
            distance = actor.get("distance", 999.0)

            if actor_type.startswith("vehicle.") and distance < self.llm_trigger_distance:
                self.llm_queried = True

                logger.info(
                    "DANGER DETECTED at %.1f m — LLM should decide brake now.",
                    distance
                )

                return True

        return False

    def _spawn_front_vehicle(self, spawn_transform):
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

        self.front_vehicle = self.world.try_spawn_actor(bp, spawn_transform)

        if self.front_vehicle is None:
            try:
                self.front_vehicle = self.world.spawn_actor(bp, spawn_transform)
            except Exception as e:
                logger.error("Failed to spawn front vehicle: %s", e)
                return

        self.npc_actors.append(self.front_vehicle)

        self.front_vehicle.set_simulate_physics(True)
        self.front_vehicle.set_autopilot(False)

        logger.info(
            "Front vehicle spawned successfully: id=%s type=%s",
            self.front_vehicle.id,
            self.front_vehicle.type_id
        )

    def teardown(self):
        if self.front_vehicle and self.front_vehicle.is_alive:
            try:
                self.front_vehicle.set_target_velocity(
                    carla.Vector3D(0.0, 0.0, 0.0)
                )

                self.front_vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0,
                        hand_brake=True
                    )
                )

            except Exception as e:
                logger.warning("Could not stop front vehicle before teardown: %s", e)

        super().teardown() 