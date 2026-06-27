from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_PROJECT_ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(_PROJECT_ROOT / ".env")

import time
import logging
import os
import argparse
from datetime import datetime

from simulation.carla_client import CarlaClient
from simulation.world_state import WorldStateExtractor
from simulation.action_executor import ActionExecutor
from simulation import step_context
from simulation.timing_config import (
    CARLA_FIXED_DELTA_S,
    NUM_STEPS,
    STEP_INTERVAL_S,
    format_step_interval_s,
    simulated_duration_s,
    ticks_per_step,
)
from evaluation.evaluator import Evaluator
from pipeline_log import log_stage, log_state_snapshot, PIPELINE

logger = logging.getLogger(__name__)


def configure_logging(level: str, log_file: str | None = None) -> str:
    """Configure logging to both the console and a txt file. Returns the path."""
    if log_file is None:
        log_dir = _PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.txt"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    level_value = getattr(logging, level.upper(), logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # utf-8 so pipeline glyphs like — and → don't crash on Windows cp1252.
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=level_value, handlers=[stream_handler, file_handler], force=True)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return str(log_file)


def main():
    parser = argparse.ArgumentParser(description="Run Agentic Driving Scenarios")
    parser.add_argument("--scenario", type=str, default="1", help="Scenario number to run (e.g., 1)")
    parser.add_argument(
        "--with-agent",
        action="store_true",
        help="Run the LLM agent for visual scenarios. By default scenarios 2 and 3 are visual-only.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="DEBUG for full world-state JSON from MCP",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.getenv("LOG_FILE"),
        help="Path to save the run log (default: logs/run_<timestamp>.txt)",
    )
    args = parser.parse_args()
    log_path = configure_logging(args.log_level, args.log_file)
    logger.info("Logging to %s", log_path)

    carla_host = os.getenv("CARLA_HOST", "127.0.0.1")
    carla_port = int(os.getenv("CARLA_PORT", 2000))
    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    num_steps = NUM_STEPS
    step_ticks = ticks_per_step()
    verbose_state = args.log_level.upper() == "DEBUG"
    scenario_only = args.scenario in {"2", "3"} and not args.with_agent

    log_stage(
        logger,
        "init",
        "CARLA %s:%s scenario=%s mode=%s llm=%s steps=%d step=%ss ticks/step=%d sim_duration=%ss",
        carla_host,
        carla_port,
        args.scenario,
        "scenario-only" if scenario_only else "agent",
        llm_provider,
        num_steps,
        format_step_interval_s(),
        step_ticks,
        int(simulated_duration_s(num_steps)),
    )

    carla_client = CarlaClient(host=carla_host, port=carla_port)
    try:
        carla_client.connect()
        carla_client.enable_synchronous_mode(fixed_delta_seconds=CARLA_FIXED_DELTA_S)
        carla_client.spawn_ego_vehicle()
        carla_client.tick()
    except Exception as e:
        logger.error("Failed to initialize CARLA simulation: %s", e)
        logger.error(
            "Restart CARLA or clear stuck vehicles at spawn, then run again."
        )
        return

    world_state = WorldStateExtractor(carla_client)
    action_executor = ActionExecutor(carla_client)

    evaluator = Evaluator(carla_client)
    if carla_client.get_ego_vehicle():
        evaluator.setup_sensors()

    if args.scenario == "1":
        from simulation.scenarios.scenario_01_braking import Scenario01Braking
        scenario = Scenario01Braking(carla_client)
    elif args.scenario == "2":
        from simulation.scenarios.scenario_02_front_vehicle_braking import (
            Scenario02FrontVehicleBraking,
        )
        scenario = Scenario02FrontVehicleBraking(carla_client)
    elif args.scenario == "3":
        from simulation.scenarios.scenario_03_pedestrian_crossing import (
            Scenario03PedestrianCrossing,
        )
        scenario = Scenario03PedestrianCrossing(carla_client)
    else:
        logger.error("Scenario %s is not implemented yet!", args.scenario)
        return

    if hasattr(scenario, "control_ego"):
        scenario.control_ego = scenario_only

    scenario.setup()
    carla_client.tick()
    if not [actor for actor in scenario.npc_actors if actor.is_alive]:
        logger.error(
            "Scenario %s produced no NPC actor — aborting run (nothing to react to). "
            "Restart CARLA or adjust the ego spawn point.",
            args.scenario,
        )
        scenario.teardown()
        evaluator.cleanup()
        carla_client.cleanup()
        return

    step_context.set_scenario_npc_ids(
        [actor.id for actor in scenario.npc_actors if actor.is_alive]
    )

    decision_maker = None
    if not scenario_only:
        try:
            from mcp_interface.server import init_mcp_server
            from mcp_interface.client import MCPDrivingClient
            from agent.llm_client import LLMClient
            from agent.decision_maker import DecisionMaker

            log_stage(logger, "init", "MCP server bridge (in-process FastMCP)")
            init_mcp_server(carla_client, world_state, action_executor)
            llm_client = LLMClient(provider=llm_provider)
            mcp_client = MCPDrivingClient(verbose_state=verbose_state)
            decision_maker = DecisionMaker(llm_client, mcp_client)
            log_stage(logger, "init", "LLM model=%s", llm_client.model)
        except Exception as e:
            logger.error("Failed to initialize LLM client: %s", e)
            scenario.teardown()
            evaluator.cleanup()
            carla_client.cleanup()
            return
    else:
        log_stage(logger, "init", "scenario-only playback: LLM agent disabled")

    log_stage(logger, "sim", "starting loop (%d steps)", num_steps)
    try:
        for step in range(num_steps):
            logger.info("%s ========== step %d/%d ==========", PIPELINE, step + 1, num_steps)

            if scenario_only:
                for tick_idx in range(step_ticks):
                    if hasattr(scenario, "update"):
                        scenario.update((step * step_ticks) + tick_idx)
                    carla_client.tick()
                    if carla_client.get_ego_vehicle():
                        spectator = carla_client.get_world().get_spectator()
                        transform = carla_client.get_ego_vehicle().get_transform()
                        import carla
                        forward_vector = transform.get_forward_vector()
                        camera_loc = transform.location - (forward_vector * 10) + carla.Location(z=5)
                        camera_rot = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
                        spectator.set_transform(carla.Transform(camera_loc, camera_rot))
                    time.sleep(CARLA_FIXED_DELTA_S)
                continue

            if hasattr(scenario, "update"):
                scenario.update(step)

            snapshot = world_state.get_state()
            log_state_snapshot(logger, snapshot, prefix="pre-step")

            step_context.begin_step(evaluator.metrics.collisions, snapshot)
            if snapshot.get("path_blocked") and not snapshot.get("maneuver_allowed"):
                log_stage(
                    logger,
                    "sim",
                    "frozen state: path_blocked but maneuver_ok=false (horizon=%sm)",
                    snapshot.get("maneuver_horizon_m"),
                )

            evaluator.metrics.start_decision_timer()
            log_stage(logger, "sim", "agent.run_step() -> LLM + MCP + CARLA")
            action = decision_maker.run_step()

            if not action:
                logger.warning("%s no action applied this step", PIPELINE)

            latency = evaluator.metrics.end_decision_timer()
            log_stage(
                logger,
                "sim",
                "step %d done | action=%s | latency=%.0fms | collisions=%d",
                step + 1,
                action,
                latency,
                evaluator.metrics.collisions,
            )

            if carla_client.get_ego_vehicle():
                spectator = carla_client.get_world().get_spectator()
                transform = carla_client.get_ego_vehicle().get_transform()
                import carla
                forward_vector = transform.get_forward_vector()
                camera_loc = transform.location - (forward_vector * 10) + carla.Location(z=5)
                camera_rot = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
                spectator.set_transform(carla.Transform(camera_loc, camera_rot))

            # Advance physics a fixed amount per step. In synchronous mode this
            # is the ONLY thing that moves the world, so a slow LLM decision can
            # no longer translate into uncontrolled travel between steps.
            if carla_client.is_synchronous():
                for tick_idx in range(step_ticks):
                    if hasattr(scenario, "update"):
                        scenario.update(
                            (step * step_ticks) + tick_idx,
                            allow_trigger=False,
                        )
                    carla_client.tick()
                    time.sleep(CARLA_FIXED_DELTA_S)
            else:
                time.sleep(STEP_INTERVAL_S)
    finally:
        # Always run teardown so synchronous mode is restored on the server even
        # if the loop raises; otherwise CARLA stays frozen for other clients.
        step_context.clear()
        log_stage(logger, "sim", "complete — teardown")
        scenario.teardown()
        evaluator.cleanup()
        carla_client.cleanup()
        evaluator.log_results()


if __name__ == "__main__":
    main()
