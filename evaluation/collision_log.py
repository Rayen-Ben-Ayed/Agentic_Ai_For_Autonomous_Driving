"""Suppress repeated collision logs while CARLA is scraping a static object."""
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

BURST_GAP_SECONDS = 2.0


class CollisionLogGate:
    def __init__(self, on_first_contact: Optional[Callable[[str], None]] = None):
        self.total_collisions = 0
        self._last_event_time: Optional[float] = None
        self._suppressed_in_burst = 0
        self._burst_actor: Optional[str] = None
        self._on_first_contact = on_first_contact
        self._burst_count = 0

    def record(self, other_actor_type: str, detail: Optional[str] = None) -> None:
        now = time.time()
        self.total_collisions += 1

        gap = (
            (now - self._last_event_time)
            if self._last_event_time is not None
            else float("inf")
        )
        new_burst = gap > BURST_GAP_SECONDS

        if new_burst:
            self._flush_burst_summary()
            self._burst_actor = other_actor_type
            self._suppressed_in_burst = 0
            self._burst_count += 1
            logger.error(
                "[collision] contact with %s (session total=%d)%s",
                other_actor_type,
                self.total_collisions,
                f" | {detail}" if detail else "",
            )
            if self._burst_count == 1 and self._on_first_contact:
                self._on_first_contact(other_actor_type)
        else:
            self._suppressed_in_burst += 1

        self._last_event_time = now

    def _flush_burst_summary(self) -> None:
        if self._suppressed_in_burst > 0:
            logger.warning(
                "[collision] +%d repeated contacts suppressed (same burst, last=%s)",
                self._suppressed_in_burst,
                self._burst_actor,
            )
        self._suppressed_in_burst = 0
        self._burst_actor = None

    def finalize(self) -> None:
        self._flush_burst_summary()
        if self.total_collisions:
            logger.info(
                "[collision] session total contacts: %d (%d bursts)",
                self.total_collisions,
                self._burst_count,
            )

    @property
    def burst_count(self) -> int:
        return self._burst_count
