from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

import logging
import os
import argparse
from datetime import datetime

from simulation.timing_config import (
    NUM_STEPS,
    format_step_interval_s,
    simulated_duration_s,
    ticks_per_step,
)
from evaluation.run_simulation import SimulationConfig, run_simulation
from pipeline_log import log_stage

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
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=level_value, handlers=[stream_handler, file_handler], force=True)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return str(log_file)


def main():
    parser = argparse.ArgumentParser(description="Run Agentic Driving Scenarios")
    parser.add_argument(
        "--scenario",
        type=str,
        default="1",
        help="Scenario number to run (1, 2, 3, 4, 5, 6, 7, or 8)",
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
    log_stage(
        logger,
        "init",
        "CARLA %s:%s scenario=%s mode=agent llm=%s steps=%d step=%ss ticks/step=%d sim_duration=%ss",
        carla_host,
        carla_port,
        args.scenario,
        llm_provider,
        num_steps,
        format_step_interval_s(),
        step_ticks,
        int(simulated_duration_s(num_steps)),
    )

    result = run_simulation(
        SimulationConfig(
            scenario=args.scenario,
            log_level=args.log_level,
            log_file=log_path,
            carla_host=carla_host,
            carla_port=carla_port,
            llm_provider=llm_provider,
        )
    )

    if result.error:
        logger.error("Run failed: %s", result.error)
        return

    log_stage(logger, "sim", "complete — collisions=%d", result.collision_events)


if __name__ == "__main__":
    main()
