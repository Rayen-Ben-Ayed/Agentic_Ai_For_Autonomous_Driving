from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
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
from simulation import telemetry
from simulation.lane_preference import enrich_keep_right_preference
from simulation.timing_config import (
    CARLA_FIXED_DELTA_S,
    NUM_STEPS,
    STEP_INTERVAL_S,
    format_step_interval_s,
    simulated_duration_s,
    ticks_per_step,
)
from mcp_interface.server import init_mcp_server
from mcp_interface.client import MCPDrivingClient
from agent.llm_client import LLMClient
from agent.decision_maker import DecisionMaker
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
    telemetry_path = telemetry.init(log_path)
    if telemetry_path:
        logger.info("Per-tick telemetry -> %s", telemetry_path)

    carla_host = os.getenv("CARLA_HOST", "127.0.0.1")
    carla_port = int(os.getenv("CARLA_PORT", 2000))
    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    num_steps = NUM_STEPS
    step_ticks = ticks_per_step()
    verbose_state = args.log_level.upper() == "DEBUG"

    log_stage(
        logger,
        "init",
        "CARLA %s:%s scenario=%s llm=%s steps=%d step=%ss ticks/step=%d sim_duration=%ss",
        carla_host,
        carla_port,
        args.scenario,
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

    log_stage(logger, "init", "MCP server bridge (in-process FastMCP)")
    init_mcp_server(carla_client, world_state, action_executor)

    try:
        llm_client = LLMClient(provider=llm_provider)
        mcp_client = MCPDrivingClient(verbose_state=verbose_state)
        decision_maker = DecisionMaker(llm_client, mcp_client)
        log_stage(logger, "init", "LLM model=%s", llm_client.model)
    except Exception as e:
        logger.error("Failed to initialize LLM client: %s", e)
        return

    evaluator = Evaluator(carla_client)
    if carla_client.get_ego_vehicle():
        evaluator.setup_sensors()

    if args.scenario == "1":
        from simulation.scenarios.scenario_01_braking import Scenario01Braking
        scenario = Scenario01Braking(carla_client)
    else:
        logger.error("Scenario %s is not implemented yet!", args.scenario)
        return

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

    log_stage(logger, "sim", "starting loop (%d steps)", num_steps)
    try:
        for step in range(num_steps):
            logger.info("%s ========== step %d/%d ==========", PIPELINE, step + 1, num_steps)

            snapshot = world_state.get_state()
            snapshot.update(action_executor.lane_centering_snapshot())
            enrich_keep_right_preference(snapshot)
            log_state_snapshot(logger, snapshot, prefix="pre-step")

            pose_before = action_executor.ego_pose_snapshot()
            if pose_before:
                log_stage(
                    logger,
                    "pre-step",
                    "ego pose x=%.2f y=%.2f yaw=%.1f lane_id=%s road_id=%s speed=%.2f",
                    pose_before.get("x"),
                    pose_before.get("y"),
                    pose_before.get("yaw"),
                    pose_before.get("lane_id"),
                    pose_before.get("road_id"),
                    pose_before.get("speed"),
                )

            action_executor.set_step_context(step + 1, step_ticks)
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
                for _ in range(step_ticks):
                    # Recompute control from the current pose along the planned
                    # trajectory, then advance one fixed-delta physics frame.
                    action_executor.tick(CARLA_FIXED_DELTA_S)
                    carla_client.tick()
            else:
                time.sleep(STEP_INTERVAL_S)

            pose_after = action_executor.ego_pose_snapshot()
            if pose_before and pose_after:
                dx = pose_after["x"] - pose_before["x"]
                dy = pose_after["y"] - pose_before["y"]
                dist = (dx * dx + dy * dy) ** 0.5
                lane_changed = pose_before.get("lane_id") != pose_after.get("lane_id")
                log_stage(
                    logger,
                    "post-step",
                    "moved dx=%.2f dy=%.2f dist=%.2fm | lane_id %s->%s (changed=%s) | "
                    "speed %.2f->%.2f | pos (%.1f,%.1f)->(%.1f,%.1f)",
                    dx,
                    dy,
                    dist,
                    pose_before.get("lane_id"),
                    pose_after.get("lane_id"),
                    lane_changed,
                    pose_before.get("speed"),
                    pose_after.get("speed"),
                    pose_before.get("x"),
                    pose_before.get("y"),
                    pose_after.get("x"),
                    pose_after.get("y"),
                )
    finally:
        # Always run teardown so synchronous mode is restored on the server even
        # if the loop raises; otherwise CARLA stays frozen for other clients.
        step_context.clear()
        log_stage(logger, "sim", "complete — teardown")
        scenario.teardown()
        evaluator.cleanup()
        carla_client.cleanup()
        evaluator.log_results()
        telemetry.close()


if __name__ == "__main__":
    main()
