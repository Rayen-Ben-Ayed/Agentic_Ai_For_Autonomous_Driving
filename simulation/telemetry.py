"""Per-run physics-tick telemetry sink.

Writes one structured JSON row per CARLA physics tick to a file next to the
main run log, giving full visibility into the otherwise-invisible tick loop
(ego pose, lane id, steering internals, control outputs). The main run log only
receives sampled human-readable lines; this file holds every tick for plotting
and post-hoc analysis.

The sink is process-global and optional: if `init()` is never called (e.g. unit
tests), `write_row()` is a no-op.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, TextIO

# Emit a sampled main-log line every Nth tick (in addition to first/last/lane
# crossings, which are always sampled).
TELEMETRY_LOG_EVERY = max(1, int(os.getenv("TELEMETRY_LOG_EVERY", "16")))

_file: Optional[TextIO] = None
_path: Optional[str] = None


def _derive_path(log_path: str) -> str:
    p = Path(log_path)
    return str(p.with_suffix(".telemetry.jsonl"))


def init(log_path: Optional[str]) -> Optional[str]:
    """Open the telemetry file next to the given log path. Returns its path."""
    global _file, _path
    close()
    if not log_path:
        return None
    if os.getenv("TELEMETRY_ENABLED", "1") not in ("1", "true", "True"):
        return None
    _path = _derive_path(log_path)
    Path(_path).parent.mkdir(parents=True, exist_ok=True)
    _file = open(_path, "w", encoding="utf-8")
    return _path


def is_active() -> bool:
    return _file is not None


def write_row(row: dict[str, Any]) -> None:
    """Append one telemetry row as a JSON line. No-op if not initialized."""
    if _file is None:
        return
    _file.write(json.dumps(row, separators=(",", ":")) + "\n")
    _file.flush()


def should_sample(tick: int, total_ticks: int, *, lane_changed: bool) -> bool:
    """Whether this tick should also produce a human-readable main-log line."""
    if lane_changed:
        return True
    if tick <= 0 or tick >= total_ticks - 1:
        return True
    return tick % TELEMETRY_LOG_EVERY == 0


def close() -> None:
    global _file, _path
    if _file is not None:
        try:
            _file.close()
        finally:
            _file = None
    _path = None
