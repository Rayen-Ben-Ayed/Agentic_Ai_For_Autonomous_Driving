# Agentic Driving

A modular Python architecture integrating CARLA, an MCP server, an LLM-based decision agent, and an evaluation pipeline.

## Project Structure

```text
agentic-driving/
├── simulation/                 # 1. CARLA Simulation Scripts
│   ├── scenarios/              # Scenario definitions (basic, interaction, edge cases)
│   ├── carla_client.py         # Connection and interaction with the CARLA server
│   ├── world_state.py          # Extracts state (ego speed, actors, obstacles) without heavy sensors
│   └── action_executor.py      # Translates discrete actions into CARLA vehicle controls
├── mcp_interface/              # 2. Model Context Protocol Bridge
│   ├── server.py               # MCP server exposing simulation state and actions
│   └── tools.py                # Defined MCP tools (e.g., get_world_state, execute_action)
├── agent/                      # 3. Agentic AI Integration
│   ├── llm_client.py           # API integration for Groq, Cerebras, and Ollama
│   ├── prompt_templates.py     # Prompts for scenario decision making
│   └── decision_maker.py       # Logic to query LLM and parse the chosen action
├── evaluation/                 # 4. Evaluation Module
│   ├── metrics.py              # Calculates collision rate, latency, rule compliance, etc.
│   └── evaluator.py            # Runs scenarios, monitors rules, and logs outcomes
├── main.py                     # Entry point to orchestrate the pipeline
└── README.md                   # Project documentation
```

## Execution Steps

1. **Start CARLA**: Run the CARLA simulator executable (`CarlaUE4.exe` or `./CarlaUE4.exe -dx11 -quality-level=Low -ResX=800 -ResY=600 -windowed`).
2. **Configure Environment**: Create a `.env` file in the project root and add your API keys (e.g., `GROQ_API_KEY=your_key`).
3. **Install Dependencies**: `pip install -r requirements.txt`
4. **Run a Scenario**: Execute the main script with the desired scenario number:
   ```bash
   python Agentic_Ai_For_Autonomous_Driving/main.py --scenario 1 
   ```
