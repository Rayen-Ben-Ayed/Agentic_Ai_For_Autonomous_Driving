import time
import logging
import os
import argparse
from dotenv import load_dotenv, find_dotenv

from simulation.carla_client import CarlaClient
from simulation.world_state import WorldStateExtractor
from simulation.action_executor import ActionExecutor
from mcp_interface.server import init_mcp_server
from agent.llm_client import LLMClient
from agent.decision_maker import DecisionMaker
from evaluation.evaluator import Evaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_scenario(scenario_number, carla_client):
    """
    Load scenario dynamically based on command line argument.
    """

    if scenario_number == "1":
        from simulation.scenarios.scenario_01_braking import Scenario01Braking
        return Scenario01Braking(carla_client)

    elif scenario_number == "2":
        from simulation.scenarios.scenario_02_front_vehicle_braking import Scenario02FrontVehicleBraking
        return Scenario02FrontVehicleBraking(carla_client)

    else:
        raise ValueError(f"Scenario {scenario_number} is not implemented yet!")


def update_spectator_camera(carla_client):
    """
    Follow ego vehicle with spectator camera.
    """

    ego_vehicle = carla_client.get_ego_vehicle()

    if not ego_vehicle:
        return

    import carla

    spectator = carla_client.get_world().get_spectator()
    transform = ego_vehicle.get_transform()
    forward = transform.get_forward_vector()

    camera_location = transform.location - forward * 12.0 + carla.Location(z=6.0)

    camera_rotation = carla.Rotation(
        pitch=-18.0,
        yaw=transform.rotation.yaw,
        roll=0.0
    )

    spectator.set_transform(
        carla.Transform(camera_location, camera_rotation)
    )


def main():
    parser = argparse.ArgumentParser(description="Run Agentic Driving Scenarios")

    parser.add_argument(
        "--scenario",
        type=str,
        default="1",
        help="Scenario number to run, e.g. 1 or 2"
    )

    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Optional CARLA map to load, e.g. Town01"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=160,
        help="Number of simulation steps"
    )

    args = parser.parse_args()

    load_dotenv(find_dotenv())

    CARLA_HOST = os.getenv("CARLA_HOST", "127.0.0.1")
    CARLA_PORT = int(os.getenv("CARLA_PORT", 2000))
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

    STEP_SLEEP = 0.05

    # 1. Initialize CARLA
    carla_client = CarlaClient(host=CARLA_HOST, port=CARLA_PORT)

    try:
        carla_client.connect()

        if args.map and hasattr(carla_client, "load_map"):
            carla_client.load_map(args.map)

        carla_client.spawn_ego_vehicle()

    except Exception as e:
        logger.error(f"Failed to initialize simulation: {e}")
        logger.error("CARLA is required. Stopping instead of running mock mode.")
        return

    world_state = WorldStateExtractor(carla_client)
    action_executor = ActionExecutor(carla_client)

    # 2. Initialize MCP Server Bridge
    init_mcp_server(carla_client, world_state, action_executor)

    # 3. Initialize LLM / decision maker
    try:
        llm_client = LLMClient(provider=LLM_PROVIDER)
        decision_maker = DecisionMaker(llm_client, mcp_server=None)

    except Exception as e:
        logger.warning(f"Failed to initialize LLM client: {e}")
        logger.warning("Running scenario without real LLM.")
        decision_maker = None

    # 4. Initialize evaluator
    evaluator = Evaluator(carla_client)

    if carla_client.get_ego_vehicle():
        evaluator.setup_sensors()

    # 5. Load scenario
    try:
        scenario = load_scenario(args.scenario, carla_client)

    except ValueError as e:
        logger.error(e)
        return

    scenario.setup()

    logger.info("Starting simulation loop...")
    logger.info("Running scenario %s for %d steps.", args.scenario, args.steps)

    # 6. Simulation loop
    for step in range(args.steps):
        logger.info(f"--- Step {step + 1}/{args.steps} ---")

        # Scenario controls normal behavior:
        # ego follows lane, NPC moves, danger is created
        if hasattr(scenario, "update"):
            scenario.update(step)

        # Get world state AFTER scenario update
        state = world_state.get_state()

        # Check whether the critical situation has appeared
        if scenario.is_llm_needed(state):
            logger.info("CRITICAL EVENT DETECTED! LLM should decide now.")

            # Important:
            # from this point, scenario should stop forcing ego throttle
            if hasattr(scenario, "give_control_to_agent"):
                scenario.give_control_to_agent()

            evaluator.metrics.start_decision_timer()

            if decision_maker:
                action = decision_maker.make_decision()
            else:
                # Temporary fake LLM decision for testing
                action = "brake"
                logger.info("No real LLM available. Using fake decision: brake")

            latency = evaluator.metrics.end_decision_timer()

            logger.info(f"Selected action: {action}")
            logger.info(f"Decision latency: {latency:.2f} ms")

            # Execute LLM / fake decision
            action_executor.execute_action(action)

        else:
            # Do NOT call action_executor.follow_lane here.
            # The scenario already controls ego movement.
            logger.info("Driving normally using scenario behavior.")

        update_spectator_camera(carla_client)

        time.sleep(STEP_SLEEP)

    # 7. Cleanup
    logger.info("Simulation complete. Cleaning up...")

    scenario.teardown()
    evaluator.cleanup()
    carla_client.cleanup()
    evaluator.log_results()


if __name__ == "__main__":
    main()