"""Live trajectory evaluation script for the ColdPulse Ops Cold-Chain Agent."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage
from src.orchestrator import get_fde_agent

SCENARIOS = [
    {
        "name": "Domino Effect (Multi-Tool Triage)",
        "prompt": (
            "Find active shipments near Los Angeles (Latitude ~33.8, Longitude ~-118.1) "
            "with Current_Temperature_C > 4.0. Check local corridor weather there and tell me "
            "if it violates the SOP for fresh perishables."
        ),
        "expected_tools": {"query_telemetry_db", "fetch_corridor_conditions", "search_compliance_sop"},
    },
    {
        "name": "Restraint (No-Telemetry Policy Query)",
        "prompt": (
            "I'm a new dispatcher on the night shift. Quickly explain the difference "
            "between a Tier 1 and Tier 2 escalation."
        ),
        "forbidden_tools": {"query_telemetry_db", "dispatch_escalation_alert"},
    },
]


def run_eval() -> int:
    agent = get_fde_agent()
    failures = 0

    for idx, scenario in enumerate(SCENARIOS, start=1):
        print(f"\n=== Scenario {idx}: {scenario['name']} ===")
        cfg = {"configurable": {"thread_id": f"eval-{idx}"}}
        result = agent.invoke({"messages": [("user", scenario["prompt"])]}, config=cfg)

        called_tools: set[str] = set()
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    called_tools.add(tc["name"])

        print(f"Tools called: {sorted(called_tools)}")

        expected = scenario.get("expected_tools", set())
        forbidden = scenario.get("forbidden_tools", set())

        missing = expected - called_tools
        violated = forbidden & called_tools
        if missing or violated:
            failures += 1
            print(f"FAIL | missing={sorted(missing)} | forbidden={sorted(violated)}")
        else:
            print("PASS")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_eval())

