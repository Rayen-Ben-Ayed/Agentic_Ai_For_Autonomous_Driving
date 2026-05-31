"""Main entrypoint for the agentic-driving pipeline (Phabmacs backend).

IMPORTANT: Start Phabmacs FIRST, then run this script.

  Terminal 1:  cd phabmacs-studi && .\\gradlew.bat run
  Terminal 2:  cd Agentic_Ai_For_Autonomous_Driving && python main.py --scenario overtake
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Immediate feedback before heavier imports
print("Agentic Driving — starting...", flush=True)

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
load_dotenv(_PROJECT_ROOT / ".env")

from simulation.phabmacs_bridge import PhabmacsBridge
from simulation.world_state import WorldStateExtractor
from simulation.action_executor import ActionExecutor
from simulation.agent_tools import init_agent_tools, execute_action
from agent.llm_client import LLMClient
from agent.decision_maker import DecisionMaker
from agent.prompt_templates import get_decision_prompt, get_system_prompt
from agent.rule_based_agent import decide_overtake
from evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def _has_llm_credentials(provider: str) -> bool:
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if provider == "cerebras":
        return bool(os.getenv("CEREBRAS_API_KEY"))
    if provider == "ollama":
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the agentic driving loop against Phabmacs",
        epilog="Start Phabmacs first: cd phabmacs-studi && gradlew run",
    )
    parser.add_argument("--host", default=os.getenv("PHABMACS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PHABMACS_PORT", "8765")))
    parser.add_argument("--steps", type=int, default=int(os.getenv("AGENT_STEPS", "40")))
    parser.add_argument("--interval", type=float, default=float(os.getenv("AGENT_INTERVAL", "1.0")))
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "groq"))
    parser.add_argument("--scenario", default=os.getenv("AGENT_SCENARIO", "overtake"))
    parser.add_argument("--results", default="evaluation_results.json")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use rule-based decisions (no LLM). Auto-enabled if API key is missing.",
    )
    parser.add_argument(
        "--bridge-wait",
        type=float,
        default=float(os.getenv("PHABMACS_BRIDGE_WAIT", "120")),
        help="Seconds to wait for Phabmacs bridge (default 120)",
    )
    args = parser.parse_args()

    use_mock = args.mock or not _has_llm_credentials(args.provider)
    if use_mock and not args.mock:
        logger.warning(
            "No API key for provider=%s — using --mock rule-based agent. "
            "Set GROQ_API_KEY in .env or pass --mock explicitly.",
            args.provider,
        )

    bridge = PhabmacsBridge(host=args.host, port=args.port)
    if not bridge.wait_until_ready(max_wait_s=args.bridge_wait):
        print(
            "\nCould not connect to Phabmacs.\n"
            "1. Open another terminal\n"
            "2. cd phabmacs-studi\n"
            "3. .\\gradlew.bat run\n"
            "4. Wait until you see: AgentBridgeServer listening on http://localhost:8765\n"
            "5. Run this script again\n",
            flush=True,
        )
        sys.exit(1)

    bridge.wait_for_ego_state(max_wait_s=90.0)

    world_state = WorldStateExtractor(bridge)
    action_executor = ActionExecutor(bridge)
    init_agent_tools(bridge, world_state, action_executor)

    decision_maker = None
    user_prompt = get_decision_prompt(args.scenario)

    if not use_mock:
        try:
            llm_client = LLMClient(provider=args.provider)
            system_prompt = get_system_prompt(args.scenario)
            decision_maker = DecisionMaker(
                llm_client, mcp_server=None, system_prompt=system_prompt
            )
            logger.info("LLM agent ready (provider=%s)", args.provider)
        except Exception as e:
            logger.error("Failed to initialize LLM: %s — falling back to --mock", e)
            use_mock = True

    evaluator = Evaluator(bridge)
    evaluator.setup_sensors()

    logger.info(
        "Decision loop: scenario=%s mode=%s steps=%d interval=%.1fs",
        args.scenario,
        "mock" if use_mock else "llm",
        args.steps,
        args.interval,
    )

    try:
        for step in range(1, args.steps + 1):
            logger.info("--- Step %d/%d ---", step, args.steps)
            state = world_state.get_state()
            traffic = state.get("traffic", {})
            logger.info(
                "Traffic: slow_ahead=%s left_clear=%s right_busy=%s dist_front=%s ego_speed=%s",
                traffic.get("slow_vehicle_ahead"),
                traffic.get("left_lane_clear"),
                traffic.get("right_lane_occupied"),
                traffic.get("distance_to_front"),
                (state.get("ego_vehicle") or {}).get("speed"),
            )

            evaluator.metrics.start_decision_timer()

            if use_mock:
                action = decide_overtake(state)
                if action:
                    execute_action(action)
            else:
                action = decision_maker.make_decision(user_message=user_prompt)

            latency = evaluator.metrics.end_decision_timer()
            logger.info("Decision: %s (latency=%.1f ms)", action, latency)

            evaluator.poll_collisions()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        evaluator.poll_collisions()
        evaluator.cleanup()
        bridge.cleanup()
        evaluator.log_results(args.results)
        logger.info("Done.")


if __name__ == "__main__":
    main()
