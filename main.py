from pathlib import Path
from dotenv import load_dotenv

# Repo .env (not cwd): find_dotenv() can miss when the shell cwd is not this project.
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

import time
import logging
import os
import argparse

from simulation.carla_client import CarlaClient
from simulation.world_state import WorldStateExtractor
from simulation.action_executor import ActionExecutor
from mcp_interface.server import init_mcp_server
from agent.llm_client import LLMClient
from agent.decision_maker import DecisionMaker
from evaluation.evaluator import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Agentic Driving Scenarios")
    parser.add_argument('--scenario', type=str, default='1', help='Scenario number to run (e.g., 1)')
    args = parser.parse_args()

    # Configuration
    CARLA_HOST = os.getenv("CARLA_HOST", "127.0.0.1")
    CARLA_PORT = int(os.getenv("CARLA_PORT", 2000))
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    NUM_STEPS = 10

    # 1. Initialize Simulation
    carla_client = CarlaClient(host=CARLA_HOST, port=CARLA_PORT)
    try:
        carla_client.connect()
        carla_client.spawn_ego_vehicle()
    except Exception as e:
        logger.error(f"Failed to initialize simulation: {e}")
        logger.info("Running in mock mode for testing without CARLA...")
        # In a real scenario, we would exit here if CARLA is required.
        # return

    world_state = WorldStateExtractor(carla_client)
    action_executor = ActionExecutor(carla_client)

    # 2. Initialize MCP Server Bridge
    init_mcp_server(carla_client, world_state, action_executor)

    # 3. Initialize Agentic AI
    try:
        llm_client = LLMClient(provider=LLM_PROVIDER)
        decision_maker = DecisionMaker(llm_client, mcp_server=None) # mcp_server is accessed globally in this skeleton
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return

    # 4. Initialize Evaluation
    evaluator = Evaluator(carla_client)
    if carla_client.get_ego_vehicle():
        evaluator.setup_sensors()

    # 5. Load Scenario dynamically based on argument
    if args.scenario == '1':
        from simulation.scenarios.scenario_01_braking import Scenario01Braking
        scenario = Scenario01Braking(carla_client)
    else:
        logger.error(f"Scenario {args.scenario} is not implemented yet!")
        return
        
    scenario.setup()

    # Run Simulation Loop
    logger.info("Starting simulation loop...")
    for step in range(NUM_STEPS):
        logger.info(f"--- Step {step + 1}/{NUM_STEPS} ---")
        
        # Get current state
        state = world_state.get_state()
        
        logger.info("Querying LLM for decision...")
        evaluator.metrics.start_decision_timer()
        
        action = decision_maker.make_decision()
        
        latency = evaluator.metrics.end_decision_timer()
        logger.info(f"Decision latency: {latency:.2f} ms")
            
        # Update spectator camera to follow the car
        if carla_client.get_ego_vehicle():
            spectator = carla_client.get_world().get_spectator()
            transform = carla_client.get_ego_vehicle().get_transform()
            # Position camera 10 meters behind and 5 meters above the car
            import carla
            forward_vector = transform.get_forward_vector()
            camera_loc = transform.location - (forward_vector * 10) + carla.Location(z=5)
            # Rotate camera to look slightly down
            camera_rot = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
            spectator.set_transform(carla.Transform(camera_loc, camera_rot))
            
        time.sleep(1.0)

    # Cleanup and Log Results
    logger.info("Simulation complete. Cleaning up...")
    scenario.teardown()
    evaluator.cleanup()
    carla_client.cleanup()
    evaluator.log_results()

if __name__ == "__main__":
    main()
