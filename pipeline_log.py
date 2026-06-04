"""Structured logging helpers for the driving agent pipeline."""
import json
import logging
from typing import Any, Optional

PIPELINE = "[pipeline]"


def summarize_world_state(state: dict, max_actors: int = 3) -> str:
    if state.get("error"):
        return f"error={state['error']}"
    ego = state.get("ego_vehicle") or {}
    parts = [
        f"speed={ego.get('speed')}",
        f"path_blocked={state.get('path_blocked')}",
        f"closest_eff={state.get('effective_closest_distance')!s}m"
        if state.get("effective_closest_distance") is not None
        else "closest_eff=—",
        *(
            [f"scenario={state.get('closest_scenario_distance')}m"]
            if state.get("closest_scenario_distance") is not None
            else []
        ),
        f"horizon={state.get('maneuver_horizon_m')}m",
        f"maneuver_ok={state.get('maneuver_allowed')}",
        f"lane_chg_ok={state.get('lane_change_allowed')}",
        f"blocking={state.get('blocking_vehicle_ahead')}",
        f"L_clear={state.get('left_lane_clear')}",
        f"R_clear={state.get('right_lane_clear')}",
        f"actors={len(state.get('nearby_actors', []))}",
    ]
    for actor in (state.get("nearby_actors") or [])[:max_actors]:
        ef = actor.get("ego_frame") or {}
        parts.append(
            f"{actor.get('type', '?')[:12]}@"
            f"lon={ef.get('longitudinal_m')} lat={ef.get('lateral_m')}m"
        )
    return " | ".join(parts)


def log_stage(logger: logging.Logger, stage: str, message: str, *args, **kwargs) -> None:
    logger.info("%s [%s] " + message, PIPELINE, stage, *args, **kwargs)


def log_state_snapshot(logger: logging.Logger, state: dict, prefix: str = "CARLA") -> None:
    log_stage(logger, prefix, summarize_world_state(state))


def log_tool_result(
    logger: logging.Logger,
    tool_name: str,
    result_text: str,
    verbose: bool = False,
) -> None:
    try:
        payload = json.loads(result_text)
    except json.JSONDecodeError:
        log_stage(logger, "MCP", "%s -> (non-JSON) %s", tool_name, result_text[:120])
        return

    if tool_name == "get_world_state":
        log_stage(logger, "MCP", "get_world_state -> %s", summarize_world_state(payload))
        if verbose:
            log_stage(logger, "MCP", "full state: %s", json.dumps(payload, indent=2))
    elif tool_name == "execute_action":
        log_stage(
            logger,
            "MCP",
            "execute_action -> status=%s action=%s msg=%s",
            payload.get("status"),
            payload.get("action") or payload.get("requested_action"),
            payload.get("message", ""),
        )
    else:
        log_stage(logger, "MCP", "%s -> %s", tool_name, payload)
