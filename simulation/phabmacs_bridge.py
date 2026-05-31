"""HTTP client that talks to the embedded Phabmacs bridge (AgentBridgeServer)."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class PhabmacsBridge:
    VALID_ACTIONS = {
        "follow_lane",
        "stop",
        "yield",
        "change_lane_left",
        "change_lane_right",
        "overtake",
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 5.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._session = requests.Session()

    def wait_until_ready(self, max_wait_s: float = 120.0, poll_s: float = 1.0) -> bool:
        """Block until the Phabmacs HTTP bridge answers /health."""
        deadline = time.time() + max_wait_s
        attempt = 0
        logger.info(
            "Waiting for Phabmacs bridge at %s (start simulator: cd phabmacs-studi && gradlew run)",
            self.base_url,
        )
        while time.time() < deadline:
            attempt += 1
            try:
                r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
                if r.status_code == 200:
                    logger.info("Connected to Phabmacs bridge at %s", self.base_url)
                    return True
            except requests.RequestException as exc:
                if attempt == 1 or attempt % 5 == 0:
                    logger.info(
                        "Still waiting for bridge (attempt %d): %s",
                        attempt,
                        exc.__class__.__name__,
                    )
            time.sleep(poll_s)

        logger.error(
            "Phabmacs bridge not reachable at %s after %.0fs. "
            "Start the simulator FIRST in another terminal:\n"
            "  cd phabmacs-studi\n"
            "  .\\gradlew.bat run\n"
            "Look for: AgentBridgeServer listening on http://localhost:8765",
            self.base_url,
            max_wait_s,
        )
        return False

    def wait_for_ego_state(self, max_wait_s: float = 90.0, poll_s: float = 0.5) -> bool:
        """Wait until /state contains a real ego snapshot (not just '{}')."""
        deadline = time.time() + max_wait_s
        logger.info("Waiting for ego vehicle state from simulator...")
        while time.time() < deadline:
            state = self.get_state()
            ego = state.get("ego_vehicle")
            if ego and ego.get("speed") is not None and "error" not in state:
                logger.info("Ego state ready (speed=%.2f m/s)", ego.get("speed"))
                return True
            time.sleep(poll_s)
        logger.warning(
            "Ego state never became ready — is the green ego vehicle spawned in Phabmacs?"
        )
        return False

    def cleanup(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def get_state(self) -> Dict[str, Any]:
        try:
            r = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                return data
            return {"error": "invalid state payload"}
        except requests.RequestException as e:
            logger.error("Failed to fetch state: %s", e)
            return {"error": str(e)}
        except ValueError as e:
            logger.error("Invalid JSON from /state: %s", e)
            return {"error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        try:
            r = self._session.get(f"{self.base_url}/metrics", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error("Failed to fetch metrics: %s", e)
            return {"collisions": 0, "current_action": "unknown", "error": str(e)}

    def send_action(self, action: str) -> bool:
        if action not in self.VALID_ACTIONS:
            logger.error("Invalid action %r (valid: %s)", action, sorted(self.VALID_ACTIONS))
            return False
        try:
            r = self._session.post(
                f"{self.base_url}/action",
                json={"action": action},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.error("send_action %s returned HTTP %s: %s", action, r.status_code, r.text)
            return False
        except requests.RequestException as e:
            logger.error("send_action %s failed: %s", action, e)
            return False

    def is_ready(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False
