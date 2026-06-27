"""Per-step shared state between main loop, MCP server, and agent."""
from __future__ import annotations

import copy
from typing import Any, Optional

_frozen_state: Optional[dict[str, Any]] = None
_collisions_at_step_start: int = 0
_live_collision_count: int = 0
_scenario_npc_ids: frozenset[int] = frozenset()
_previewed_actions: set[str] = set()


def begin_step(collision_count: int, live_state: dict[str, Any]) -> None:
    global _frozen_state, _collisions_at_step_start, _live_collision_count
    _collisions_at_step_start = collision_count
    _live_collision_count = collision_count
    _frozen_state = copy.deepcopy(live_state)
    _previewed_actions.clear()


def set_scenario_npc_ids(actor_ids: list[int]) -> None:
    global _scenario_npc_ids
    _scenario_npc_ids = frozenset(actor_ids)


def get_scenario_npc_ids() -> frozenset[int]:
    return _scenario_npc_ids


def clear() -> None:
    global _frozen_state, _collisions_at_step_start, _live_collision_count, _scenario_npc_ids
    _frozen_state = None
    _collisions_at_step_start = 0
    _live_collision_count = 0
    _scenario_npc_ids = frozenset()
    _previewed_actions.clear()


def record_preview(action: str) -> None:
    """Mark an action as previewed (cross-checked) during the current step."""
    _previewed_actions.add(action)


def was_previewed(action: str) -> bool:
    return action in _previewed_actions


def get_frozen_state() -> Optional[dict[str, Any]]:
    return _frozen_state


def collisions_at_step_start() -> int:
    return _collisions_at_step_start


def update_live_collision_count(count: int) -> None:
    global _live_collision_count
    _live_collision_count = count


def collisions_this_step() -> int:
    return max(0, _live_collision_count - _collisions_at_step_start)


def collisions_increased_this_step() -> bool:
    return collisions_this_step() > 0
