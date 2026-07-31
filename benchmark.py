#!/usr/bin/env python3
"""Benchmark harness for agentic driving scenarios."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

from agent.llm_client import LLMClient
from evaluation.benchmark_collector import BenchmarkCollector
from evaluation.benchmark_runner import (
    aggregate_runs,
    print_benchmark_summary,
    save_benchmark_report,
)
from evaluation.run_simulation import SimulationConfig, run_simulation
from main import configure_logging

logger = logging.getLogger(__name__)

AVAILABLE_SCENARIOS = ("1", "2", "3", "4", "5", "6", "7", "8")


def _parse_scenarios(raw: str) -> list[str]:
    if raw.lower() == "all":
        return list(AVAILABLE_SCENARIOS)
    scenarios = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [s for s in scenarios if s not in AVAILABLE_SCENARIOS]
    if invalid:
        raise ValueError(
            f"Unknown scenario(s): {', '.join(invalid)}. "
            f"Choose from {', '.join(AVAILABLE_SCENARIOS)} or 'all'."
        )
    return scenarios


def _resolve_llm_model(provider: str | None) -> str | None:
    try:
        client = LLMClient(provider=provider)
        return client.model
    except Exception:
        return None


def run_benchmark_for_scenario(
    scenario: str,
    repeats: int,
    *,
    log_level: str,
    output: str | None,
    pause_s: float,
    llm_provider: str | None,
) -> str:
    llm_model = _resolve_llm_model(llm_provider)
    runs = []

    for run_idx in range(1, repeats + 1):
        log_dir = _PROJECT_ROOT / "logs" / "benchmark"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / (
            f"s{scenario}_run{run_idx}_{datetime.now():%Y%m%d_%H%M%S}.txt"
        )
        configure_logging(log_level, str(log_file))

        collector = BenchmarkCollector()
        logger.info(
            "Benchmark run %d/%d — scenario %s (log: %s)",
            run_idx,
            repeats,
            scenario,
            log_file,
        )

        result = run_simulation(
            SimulationConfig(
                scenario=scenario,
                log_level=log_level,
                log_file=str(log_file),
                llm_provider=llm_provider,
                benchmark_collector=collector,
                run_index=run_idx,
            )
        )
        runs.append(result)

        if result.error:
            logger.error("Run %d failed: %s", run_idx, result.error)
        else:
            logger.info(
                "Run %d complete — collisions=%d actions=%s",
                run_idx,
                result.collision_events,
                result.action_sequence,
            )

        if run_idx < repeats and pause_s > 0:
            logger.info("Waiting %.1fs before next run...", pause_s)
            time.sleep(pause_s)

    report = aggregate_runs(
        scenario,
        repeats,
        runs,
        llm_provider=llm_provider or os.getenv("LLM_PROVIDER", "groq"),
        llm_model=llm_model,
    )
    report_path = save_benchmark_report(report, output)
    print_benchmark_summary(report)
    print(f"Full report saved to: {report_path}")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark agentic driving scenarios (repeatable runs + metrics)."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario id(s) to run: e.g. 3, or 3,7,8, or all",
    )
    parser.add_argument(
        "--repeats",
        "-n",
        type=int,
        default=1,
        help="Number of times to run each selected scenario (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for JSON benchmark report (default: benchmark_results/...)",
    )
    parser.add_argument(
        "--pause-between-runs",
        type=float,
        default=float(os.getenv("BENCHMARK_PAUSE_S", "2.0")),
        help="Seconds to wait between repeated runs (default: 2.0)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=os.getenv("LLM_PROVIDER"),
        help="Override LLM_PROVIDER for this benchmark session",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    try:
        scenarios = _parse_scenarios(args.scenario)
    except ValueError as exc:
        parser.error(str(exc))

    exit_code = 0
    for scenario in scenarios:
        scenario_output = args.output
        if scenario_output and len(scenarios) > 1:
            out = Path(scenario_output)
            scenario_output = str(
                out.with_name(f"{out.stem}_s{scenario}{out.suffix or '.json'}")
            )
        try:
            run_benchmark_for_scenario(
                scenario,
                args.repeats,
                log_level=args.log_level,
                output=scenario_output,
                pause_s=args.pause_between_runs,
                llm_provider=args.llm_provider,
            )
        except Exception:
            logger.exception("Benchmark failed for scenario %s", scenario)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
