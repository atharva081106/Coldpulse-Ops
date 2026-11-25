# ColdPulse Ops — Cold-Chain Logistics Dispatch Console

An agentic **LangGraph + Streamlit** operations platform for monitoring refrigerated fleet telemetry, evaluating corridor weather risks, enforcing Standard Operating Procedures (SOPs), and dispatching **Human-in-the-Loop (HITL)** escalations.

## Key Capabilities
- **Dual-Mode Data Backend (`src/db_backend.py`)**:
  - **Local Demo Mode (`DATA_BACKEND=LOCAL` / zero-config fallback)**: Automatically seeds a local SQLite database (`data/cache/local_fleet.db`) from `data/raw/dynamic_supply_chain_logistics_dataset.csv` and runs local TF-IDF SOP retrieval over `data/policy/`.
  - **Enterprise Mode (`DATA_BACKEND=MSSQL`)**: Connects to SQL Server (`FDE_VIEWS.VW_ACTIVE_FLEET` & `AgentAuditLog`) with strict read-only SQL guardrails and Pinecone vector search.
- **Hardened Agent Tools (`src/agent_tools.py`)**:
  - `query_telemetry_db`: Single-statement `SELECT`/`WITH` validation, pooled connections, Markdown table formatting.
  - `fetch_corridor_conditions`: Open-Meteo live weather + 6-hour precipitation forecast + ambient heat/freeze reefer stress scoring.
  - `search_compliance_sop`: Pinecone RAG with automatic local Markdown SOP fallback.
  - `dispatch_escalation_alert`: Logs escalation tickets and optionally notifies `ALERT_WEBHOOK_URL`.
- **LangGraph Orchestrator with HITL (`src/orchestrator.py`)**:
  - Automatically injects `src/prompts/system_prompt.txt` into every session.
  - Pauses before `action_tools` (`interrupt_before=["action_tools"]`) so a human dispatcher must approve or reject escalation dispatches.
- **Command Center UI (`src/ui.py`)**:
  - Fleet KPI bar, interactive telemetry map, one-click anomaly scan buttons, HITL approval cards, Escalation Tickets view, and Audit Log inspector.

## Quickstart (Local Zero-Config Mode)
1. Copy `.env.example` to `.env` and set your LLM key (e.g. `Agent_llm=OPENAI` and `OPENAI_API_KEY=...`, or use local `OLLAMA`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tests:
   ```bash
   pytest -v
   ```
4. Launch the dashboard:
   ```bash
   streamlit run src/ui.py
   ```

