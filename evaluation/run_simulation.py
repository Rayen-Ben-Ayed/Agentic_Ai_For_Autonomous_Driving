"""Single CARLA simulation run, shared by main.py and the benchmark harness."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    ticks_per_step,
)
from mcp_interface.server import init_mcp_server
from mcp_interface.client import MCPDrivingClient
from agent.llm_client import LLMClient
from agent.decision_maker import DecisionMaker
from evaluation.evaluator import Evaluator
from evaluation.benchmark_collector import BenchmarkCollector, RunBenchmarkResult
from pipeline_log import log_stage, log_state_snapshot, PIPELINE

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SimulationConfig:
    scenario: str
    with_agent: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    carla_host: Optional[str] = None
    carla_port: Optional[int] = None
    llm_provider: Optional[str] = None
    benchmark_collector: Optional[BenchmarkCollector] = None
    run_index: int = 1


def load_scenario(scenario_id: str, carla_client):
    if scenario_id == "1":
        from simulation.scenarios.scenario_01_braking import Scenario01Braking

        return Scenario01Braking(carla_client)
    if scenario_id == "4":
        from simulation.scenarios.scenario_04_multi_car_braking import (
            Scenario04MultiCarBraking,
        )

        return Scenario04MultiCarBraking(carla_client)
    if scenario_id == "5":
        from simulation.scenarios.scenario_05_multi_car_pedestrian import (
            Scenario05MultiCarPedestrian,
        )

        return Scenario05MultiCarPedestrian(carla_client)
    if scenario_id == "2":
        from simulation.scenarios.scenario_02_front_vehicle_braking import (
            Scenario02FrontVehicleBraking,
        )

        return Scenario02FrontVehicleBraking(carla_client)
    if scenario_id == "3":
        from simulation.scenarios.scenario_03_pedestrian_crossing import (
            Scenario03PedestrianCrossing,
        )

        return Scenario03PedestrianCrossing(carla_client)
    if scenario_id == "6":
        from simulation.scenarios.scenario_06_right_lane_pullout import (
            Scenario06RightLanePullout,
        )

        return Scenario06RightLanePullout(carla_client)
    if scenario_id == "7":
        from simulation.scenarios.scenario_07_blocked_lane_clear_left import (
            Scenario07BlockedLaneClearLeft,
        )

        return Scenario07BlockedLaneClearLeft(carla_client)
    if scenario_id == "8":
        from simulation.scenarios.scenario_08_blocked_lane_unsafe_left import (
            Scenario08BlockedLaneUnsafeLeft,
        )

        return Scenario08BlockedLaneUnsafeLeft(carla_client)
    return None


def _follow_spectator(carla_client) -> None:
    if not carla_client.get_ego_vehicle():
        return
    import carla

    spectator = carla_client.get_world().get_spectator()
    transform = carla_client.get_ego_vehicle().get_transform()
    forward_vector = transform.get_forward_vector()
    camera_loc = transform.location - (forward_vector * 10) + carla.Location(z=5)
    camera_rot = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
    spectator.set_transform(carla.Transform(camera_loc, camera_rot))


def run_simulation(config: SimulationConfig) -> RunBenchmarkResult:
    """Execute one scenario run. Returns benchmark metrics even on failure."""
    collector = config.benchmark_collector
    if collector is not None:
        collector.activate()

    result = RunBenchmarkResult(
        scenario=config.scenario,
        run_index=config.run_index,
        success=False,
        collision_events=0,
        total_contact_substeps=0,
        first_collision_with=None,
        rule_violations=0,
        log_file=config.log_file,
    )

    carla_host = config.carla_host or os.getenv("CARLA_HOST", "127.0.0.1")
    carla_port = config.carla_port or int(os.getenv("CARLA_PORT", 2000))
    llm_provider = config.llm_provider or os.getenv("LLM_PROVIDER", "groq")
    num_steps = NUM_STEPS
    step_ticks = ticks_per_step()
    scenario_only = config.scenario in {"2", "3", "6", "7", "8"} and not config.with_agent

    if config.log_file:
        telemetry_path = telemetry.init(config.log_file)
        if telemetry_path:
            logger.info("Per-tick telemetry -> %s", telemetry_path)

    if scenario_only and collector is not None:
        result.error = "Benchmark requires agent mode; use --with-agent for this scenario."
        return result

    carla_client = CarlaClient(host=carla_host, port=carla_port)
    evaluator: Optional[Evaluator] = None
    scenario = None

    try:
        carla_client.connect()
        scenario_spawn_point = None
        if config.scenario in {"6", "7", "8"}:
            from simulation.scenarios.scenario_07_blocked_lane_clear_left import (
                SCENARIO_MAPS,
                select_midroad_spawn_point,
            )

            target_map = SCENARIO_MAPS.get(config.scenario)
            if target_map and target_map not in carla_client.get_world().get_map().name:
                logger.info(
                    "Scenario %s loading %s for a cleaner mid-road setup.",
                    config.scenario,
                    target_map,
                )
                carla_client.world = carla_client.client.load_world(target_map)
                time.sleep(1.0)
            if config.scenario == "6":
                from simulation.scenarios.scenario_06_right_lane_pullout import (
                    select_right_lane_pullout_spawn_point,
                )

                scenario_spawn_point = select_right_lane_pullout_spawn_point(
                    carla_client.get_world()
                )
            else:
                scenario_spawn_point = select_midroad_spawn_point(
                    carla_client.get_world(),
                    variant=config.scenario,
                )
        carla_client.enable_synchronous_mode(fixed_delta_seconds=CARLA_FIXED_DELTA_S)
        carla_client.spawn_ego_vehicle(spawn_point=scenario_spawn_point)
        carla_client.tick()

        world_state = WorldStateExtractor(carla_client)
        action_executor = ActionExecutor(carla_client)
        evaluator = Evaluator(carla_client)
        if carla_client.get_ego_vehicle():
            evaluator.setup_sensors()

        scenario = load_scenario(config.scenario, carla_client)
        if scenario is None:
            result.error = f"Scenario {config.scenario} is not implemented."
            return result

        if hasattr(scenario, "control_ego"):
            scenario.control_ego = scenario_only

        scenario.setup()
        carla_client.tick()
        if not [actor for actor in scenario.npc_actors if actor.is_alive]:
            result.error = (
                f"Scenario {config.scenario} produced no NPC actor — "
                "restart CARLA or adjust spawn."
            )
            return result

        step_context.set_scenario_npc_ids(
            [actor.id for actor in scenario.npc_actors if actor.is_alive]
        )

        decision_maker = None
        if not scenario_only:
            init_mcp_server(carla_client, world_state, action_executor)
            llm_client = LLMClient(
                provider=llm_provider,
                benchmark_collector=collector,
            )
            mcp_client = MCPDrivingClient(
                verbose_state=config.log_level.upper() == "DEBUG",
                benchmark_collector=collector,
            )
            decision_maker = DecisionMaker(
                llm_client,
                mcp_client,
                benchmark_collector=collector,
            )

        for step in range(num_steps):
            logger.info("%s ========== step %d/%d ==========", PIPELINE, step + 1, num_steps)

            if scenario_only:
                for tick_idx in range(step_ticks):
                    if hasattr(scenario, "update"):
                        scenario.update((step * step_ticks) + tick_idx)
                    carla_client.tick()
                    _follow_spectator(carla_client)
                continue

            if hasattr(scenario, "update"):
                scenario.update(step)

            snapshot = world_state.get_state()
            snapshot.update(action_executor.lane_centering_snapshot())
            snapshot.update(action_executor.junction_snapshot())
            enrich_keep_right_preference(snapshot)
            log_state_snapshot(logger, snapshot, prefix="pre-step")

            action_executor.set_step_context(step + 1, step_ticks)
            step_context.begin_step(evaluator.metrics.collisions, snapshot)

            if collector is not None:
                collector.begin_decision(step + 1)

            action = decision_maker.run_step()
            if collector is not None:
                collector.end_decision(action)

            if not action:
                logger.warning("%s no action applied this step", PIPELINE)

            _follow_spectator(carla_client)

            if carla_client.is_synchronous():
                for tick_idx in range(step_ticks):
                    if hasattr(scenario, "update"):
                        scenario.update(
                            (step * step_ticks) + tick_idx,
                            allow_trigger=False,
                        )
                    action_executor.tick(CARLA_FIXED_DELTA_S)
                    carla_client.tick()
            else:
                time.sleep(STEP_INTERVAL_S)

        result.success = True
    except Exception as exc:
        logger.exception("Simulation run failed")
        result.error = str(exc)
    finally:
        step_context.clear()
        if scenario is not None:
            scenario.teardown()
        if evaluator is not None:
            evaluator.cleanup()
        carla_client.cleanup()

        if evaluator is not None:
            summary = evaluator.metrics.get_summary()
            result.collision_events = evaluator.collision_log.burst_count
            result.total_contact_substeps = summary.get("total_contact_substeps", 0)
            result.first_collision_with = summary.get("first_collision_with")
            result.rule_violations = summary.get("rule_violations", 0)
            results_path = "evaluation_results.json"
            if config.log_file:
                results_path = str(Path(config.log_file).with_suffix(".evaluation.json"))
            evaluator.log_results(results_path)

        if collector is not None:
            result.decisions = list(collector.decisions)
            result.action_sequence = list(collector.action_sequence)

        telemetry.close()

    return result
