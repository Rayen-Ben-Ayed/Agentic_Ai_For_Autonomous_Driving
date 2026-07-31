# eAgentic AI for Autonomous Driving

An LLM-driven driving agent for the [CARLA](https://carla.org/) simulator. The system couples a synchronous CARLA world, an MCP (Model Context Protocol) tool bridge, runtime safety guards, and a benchmark harness for repeatable evaluation across eight scripted scenarios.

## Architecture

```text
┌─────────────┐     MCP tools      ┌──────────────────┐     vehicle controls
│  LLM Agent  │ ◄───────────────► │  MCP Server      │ ◄──────────────────► CARLA
│ (Groq, etc.)│  get_world_state  │  + maneuver      │
│             │  execute_action   │    policy gates  │
└─────────────┘                   └──────────────────┘
        ▲                                   ▲
        │                                   │
   decision_maker.py                  action_executor.py
                                      world_state.py
```

Each simulation step:

1. The agent reads structured world state via MCP (`get_world_state`).
2. The LLM chooses a discrete action and calls `execute_action`.
3. Runtime guards filter infeasible or unsafe actions before execution.
4. The action executor applies low-level controls for the configured step interval.
5. CARLA advances in synchronous mode (fixed physics delta per tick).

## Requirements

- **Python** 3.10+ (MCP package requires 3.10+)
- **CARLA** 0.9.x simulator (tested with synchronous mode on Town03–Town10 maps)
- An API key for at least one supported LLM provider (see [Configuration](#configuration))

## Quick start

### 1. Start CARLA

Launch the CARLA server before running any scenario, for example:

```bash
CarlaUE4.exe -quality-level=Low -ResX=800 -ResY=600 -windowed
```

### 2. Install dependencies

```bash
cd Agentic_Ai_For_Autonomous_Driving
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

> **Note:** The `carla` Python package must match your CARLA simulator build. Install it from the CARLA distribution's `PythonAPI/carla/dist/` wheel if `pip install carla` does not match your version.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- `LLM_PROVIDER` — one of `groq`, `cerebras`, `ollama`, `ollama-remote`, `academic_cloud`
- The matching API key (e.g. `GROQ_API_KEY`)

All timing, perception, and control thresholds are documented in `.env.example`.

### 4. Run a scenario

```bash
python main.py --scenario 1
```

Optional flags:

```bash
python main.py --scenario 3 --log-level DEBUG --log-file logs/my_run.txt
```

Logs are written to `logs/run_<timestamp>.txt` by default. When telemetry is enabled, a companion `.telemetry.jsonl` file is created next to the log.

## Scenarios


| ID  | Name                       | Map          | Description                                        |
| --- | -------------------------- | ------------ | -------------------------------------------------- |
| 1   | Emergency braking          | Town03       | Stopped vehicle ahead on the ego lane              |
| 2   | Crossing vehicle           | Town03       | Vehicle crosses the ego path from the side         |
| 3   | Pedestrian crossing        | Town03       | Pedestrian crosses ahead at a trigger distance     |
| 4   | Multi-car braking          | Town10HD_Opt | Slow lead vehicle with surrounding traffic         |
| 5   | Multi-car + pedestrian     | Town10HD_Opt | Scenario 4 plus a crossing pedestrian              |
| 6   | Right-lane pullout         | Town10HD_Opt | Left-lane vehicle merges into ego lane             |
| 7   | Blocked lane (safe left)   | Town04       | Stopped vehicle ahead; left lane is clear          |
| 8   | Blocked lane (unsafe left) | Town05       | Right and middle lanes blocked; left lane occupied |


Scenarios 6–8 load dedicated maps and custom spawn points automatically.

## Action space

The agent selects from eight discrete actions:

`follow_lane`, `stop`, `yield`, `change_lane_left`, `change_lane_right`, `go_straight`, `turn_left`, `turn_right`

Before execution, the maneuver policy computes `allowed_actions` from simulator ground truth (obstacles, lane geometry, pedestrian conflicts, junction context). The MCP server rejects actions outside that set.

## Benchmarking

Run repeatable multi-run evaluations with aggregated metrics:

```bash
# Single scenario, 5 repeats
python benchmark.py --scenario 1 --repeats 5

# Multiple scenarios
python benchmark.py --scenario 3,7,8 --repeats 5

# All scenarios
python benchmark.py --scenario all --repeats 3
```

Reports are saved as JSON under `benchmark_results/` (collision rate, decision latency, token usage, action acceptance, cross-run determinism). Example reports from the evaluation campaign are included:

- `benchmark_results/benchmark_scenario_01_run_01.json` … `benchmark_scenario_08_run_01.json`

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

Unit tests cover maneuver policy, lane controllers, timing configuration, MCP agent logic, and benchmark aggregation. CARLA is not required for the test suite.

## Project structure

```text
Agentic_Ai_For_Autonomous_Driving/
├── agent/                  # LLM client, prompts, decision loop
│   ├── llm_client.py
│   ├── llm_config.py
│   ├── decision_maker.py
│   └── prompt_templates.py
├── mcp_interface/          # MCP server and client bridge
│   ├── server.py
│   └── client.py
├── simulation/             # CARLA integration
│   ├── carla_client.py
│   ├── world_state.py
│   ├── action_executor.py
│   ├── maneuver_policy.py
│   ├── timing_config.py
│   └── scenarios/          # Scenario 1–8 definitions
├── evaluation/             # Metrics, evaluator, benchmark harness
│   ├── run_simulation.py
│   ├── benchmark_runner.py
│   └── evaluator.py
├── tests/                  # Unit tests
├── benchmark_results/      # Saved benchmark JSON reports
├── main.py                 # Single-scenario entry point
├── benchmark.py            # Multi-run benchmark entry point
├── .env.example            # Configuration template (copy to .env)
└── requirements.txt
```

## Configuration reference

Key environment variables (full list in `.env.example`):


| Variable              | Default | Purpose                                     |
| --------------------- | ------- | ------------------------------------------- |
| `LLM_PROVIDER`        | `groq`  | LLM backend                                 |
| `STEP_INTERVAL_S`     | `4.0`   | Simulated seconds per agent decision        |
| `NUM_STEPS`           | `16`    | Agent decisions per scenario run            |
| `CARLA_FIXED_DELTA_S` | `0.05`  | Physics sub-step in synchronous mode        |
| `TELEMETRY_ENABLED`   | `1`     | Per-tick JSONL telemetry alongside run logs |


Timing values consumed by prompts and maneuver planning are centralized in `simulation/timing_config.py`.

## Supported LLM providers


| Provider        | Env key             | Notes                                                   |
| --------------- | ------------------- | ------------------------------------------------------- |
| Groq            | `GROQ_API_KEY`      | OpenAI-compatible API                                   |
| Cerebras        | `CEREBRAS_API_KEY`  | OpenAI-compatible API                                   |
| Ollama (local)  | —                   | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) |
| Ollama (remote) | —                   | `OLLAMA_REMOTE_BASE_URL`                                |
| Academic Cloud      | `ACADEMIC_CLOUD_API_KEY` |  endpoint                                   |


