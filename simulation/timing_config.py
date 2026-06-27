"""Single source of truth for simulation run timing (read from environment)."""
from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# Simulated seconds between agent decisions; each action is committed for this window.
STEP_INTERVAL_S = _float_env("STEP_INTERVAL_S", 4.0)

# Extra reaction time folded into maneuver horizon (0 in synchronous mode).
AGENT_LATENCY_S = _float_env("AGENT_LATENCY_S", 0.0)

# CARLA fixed physics sub-step; world advances only on client ticks.
CARLA_FIXED_DELTA_S = _float_env("CARLA_FIXED_DELTA_S", 0.05)

# Agent decisions per scenario run (one get_world_state + execute_action cycle each).
NUM_STEPS = max(1, _int_env("NUM_STEPS", 10))

# Max LLM tool-call rounds within a single step before forcing stop.
MAX_LLM_TOOL_ROUNDS = max(1, _int_env("MAX_LLM_TOOL_ROUNDS", 6))


def ticks_per_step() -> int:
    """Physics ticks executed after each agent decision."""
    return max(1, round(STEP_INTERVAL_S / CARLA_FIXED_DELTA_S))


def simulated_duration_s(num_steps: int | None = None) -> float:
    """Total simulated seconds for a run with `num_steps` agent decisions."""
    steps = NUM_STEPS if num_steps is None else num_steps
    return steps * STEP_INTERVAL_S


def format_step_interval_s() -> str:
    """Human-readable step length for logs and prompts."""
    if STEP_INTERVAL_S == int(STEP_INTERVAL_S):
        return str(int(STEP_INTERVAL_S))
    return f"{STEP_INTERVAL_S:g}"
