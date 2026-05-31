# Agentic Driving (Phabmacs backend)

A modular architecture for an LLM-driven autonomous-driving agent on top of the
**Phabmacs** simulator developed at DCAITI / TU Berlin. The agent receives a
discrete world state from Phabmacs, asks an LLM (Groq, Cerebras or Ollama) for a
maneuver, and executes one of six discrete actions back in the simulator.

```
agentic-driving/
├── phabmacs-studi/                       # Kotlin/Java Phabmacs scenario + bridge
│   └── src/main/kotlin/.../apps/
│       ├── AgentAppSim.kt                # Ego + traffic + collision listener
│       └── agent/
│           ├── AgentDrivingSkill.kt      # Custom Skill controlling the ego
│           └── AgentBridgeServer.kt      # Embedded HTTP server (port 8765)
└── Agentic_Ai_For_Autonomous_Driving/    # Python agent + MCP + evaluation
    ├── simulation/
    │   ├── phabmacs_bridge.py            # HTTP client to AgentBridgeServer
    │   ├── world_state.py                # Reshapes JSON for the prompt
    │   └── action_executor.py            # Posts discrete actions
    ├── mcp_interface/server.py           # MCP tools: get_world_state / execute_action
    ├── agent/
    │   ├── llm_client.py                 # Groq / Cerebras / Ollama
    │   ├── prompt_templates.py
    │   └── decision_maker.py
    ├── evaluation/
    │   ├── metrics.py                    # Decision latency, collisions, violations
    │   └── evaluator.py                  # Polls /metrics on the bridge
    └── main.py                           # Decision loop entrypoint
```

## How the two halves talk

| Endpoint                 | Direction        | Purpose                                                |
| ------------------------ | ---------------- | ------------------------------------------------------ |
| `GET  /health`           | Python -> Phabmacs | Readiness probe                                       |
| `GET  /state`            | Python -> Phabmacs | Ego speed/pose, front vehicle, 8-way surroundings     |
| `POST /action`           | Python -> Phabmacs | Body `{"action":"..."}`; queued for next sim step      |
| `GET  /metrics`          | Python -> Phabmacs | Collision counter + current applied action            |

The six discrete actions are mapped inside `AgentDrivingSkill.applyAction`:

| Agent action        | Phabmacs effect                                                   |
| ------------------- | ----------------------------------------------------------------- |
| `follow_lane`       | Reset to baseline max speed and headway                           |
| `stop`              | `setMaximumSpeed(0)`, increased headway                           |
| `yield`             | Max speed = 40% of baseline, large headway                        |
| `change_lane_left`  | `LaneChangeFeature.requestLaneChangeLeft()`                       |
| `change_lane_right` | `LaneChangeFeature.requestLaneChangeRight()`                      |
| `overtake`          | Lane change left + 120% baseline speed + smaller headway          |

## Running

**Order matters:** Phabmacs must be running before `python main.py`, or the script will
wait up to 2 minutes and exit with instructions.

### 1. Start the simulator (overtake scenario)

```powershell
cd phabmacs-studi
.\gradlew.bat run
```

This runs `OvertakeScenarioAppSim`: green ego on the center lane approaches a
slow red car ahead; an amber car occupies the right lane; the left lane stays
empty for a safe overtake.

The Phabmacs window opens and the console prints:

```
AgentBridgeServer listening on http://localhost:8765
```

### 2. Configure the LLM

Inside `Agentic_Ai_For_Autonomous_Driving/.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=...
```

### 3. Start the agent

```powershell
cd Agentic_Ai_For_Autonomous_Driving
pip install -r requirements.txt
python main.py --scenario overtake --steps 40 --interval 1.0
```

Use `--scenario default` with `AgentAppSim` if you switch `build.gradle.kts`
`mainClass` back to `AgentAppSimKt`.

### Troubleshooting: "nothing happens"

| Symptom | Fix |
|--------|-----|
| Python prints nothing for a long time | Start Phabmacs first (`gradlew run`). You should see `Waiting for Phabmacs bridge...` every few seconds. |
| `Could not connect to Phabmacs` | Confirm console shows `AgentBridgeServer listening on http://localhost:8765`. |
| Python runs but the car does not move | The sim must be in the foreground and unpaused; actions apply on the next physics step. |
| `No response from LLM` | Add `GROQ_API_KEY` to `.env`, or run `python main.py --mock` (rule-based, no API). |
| `dist_front=None` in logs | Wait a few steps until ego gets close to the red car; respawn by restarting Phabmacs. |

Quick test without an API key:

```powershell
python main.py --mock --steps 20
```

The script:

1. Polls `/health` until the simulator is ready.
2. Each step queries `/state`, sends it to the LLM via the MCP tools, and POSTs
   the chosen action back to `/action`.
3. Polls `/metrics` for collisions and writes a JSON summary on exit.

Flags:

- `--host` / `--port`     – override bridge address (default `localhost:8765`)
- `--steps`               – number of decision steps (default 30)
- `--interval`            – seconds between decisions (default 1.0)
- `--provider`            – `groq` / `cerebras` / `ollama`
- `--results`             – output JSON for the evaluator

## Migration from the CARLA prototype

The old `simulation/carla_client.py` and CARLA-specific scenarios have been
replaced by the Phabmacs bridge. The agent layer (`agent/`, MCP tools, metrics)
is unchanged in interface but now talks HTTP instead of the CARLA RPC.
