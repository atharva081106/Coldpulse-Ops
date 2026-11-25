import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.messages import AIMessage, HumanMessage

os.environ["DATA_BACKEND"] = "LOCAL"
os.environ["SOP_BACKEND"] = "LOCAL"

from src.agent_tools import (
    fetch_corridor_conditions,
    query_telemetry_db,
    search_compliance_sop,
)
from src.db_backend import _translate_tsql_to_sqlite, validate_readonly_sql
from src.orchestrator import build_agent


def test_sql_guardrail_allows_select_and_cte():
    ok, _ = validate_readonly_sql("SELECT TOP 5 Latitude, Longitude FROM FDE_VIEWS.VW_ACTIVE_FLEET")
    assert ok is True

    ok_cte, _ = validate_readonly_sql("WITH cte AS (SELECT * FROM FDE_VIEWS.VW_ACTIVE_FLEET) SELECT * FROM cte")
    assert ok_cte is True


@pytest.mark.parametrize(
    "bad_sql",
    [
        "",
        "DELETE FROM FDE_VIEWS.VW_ACTIVE_FLEET",
        "SELECT * FROM FDE_VIEWS.VW_ACTIVE_FLEET; DROP TABLE AgentAuditLog;",
        "SELECT * INTO dbo.TempCopy FROM FDE_VIEWS.VW_ACTIVE_FLEET",
        "UPDATE FDE_VIEWS.VW_ACTIVE_FLEET SET Current_Temperature_C = 0",
        "EXEC xp_cmdshell 'whoami'",
    ],
)
def test_sql_guardrail_blocks_mutations(bad_sql: str):
    ok, msg = validate_readonly_sql(bad_sql)
    assert ok is False
    assert "SECURITY BLOCK" in msg


def test_tsql_translation_top_and_schema():
    sql = "SELECT TOP 3 Latitude, Longitude FROM FDE_VIEWS.VW_ACTIVE_FLEET ORDER BY Timestamp DESC"
    translated = _translate_tsql_to_sqlite(sql)
    assert "FDE_VIEWS." not in translated
    assert "TOP 3" not in translated
    assert translated.endswith("LIMIT 3")


def test_local_telemetry_query_returns_markdown_table():
    res = query_telemetry_db.invoke(
        "SELECT TOP 2 Latitude, Longitude, Current_Temperature_C FROM FDE_VIEWS.VW_ACTIVE_FLEET"
    )
    assert "Returned 2 row(s):" in res
    assert "| Latitude | Longitude | Current_Temperature_C |" in res


def test_local_sop_retrieval():
    res = search_compliance_sop.invoke("What are the temperature rules for fresh perishables?")
    assert "COMPLIANCE SOP CONTEXT" in res
    assert "4.0" in res

    pharma = search_compliance_sop.invoke("What is the temperature range for pharma vaccines?")
    assert "2.0" in pharma and "8.0" in pharma

    frozen = search_compliance_sop.invoke("What is the threshold for frozen seafood?")
    assert "-18.0" in frozen


def test_corridor_conditions_multifactor_risk():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "current_weather": {"temperature": 35.0, "windspeed": 40.0, "weathercode": 63},
        "hourly": {"precipitation_probability": [80, 75, 50, 20, 10, 5]},
    }
    mock_resp.raise_for_status.return_value = None

    with patch("src.agent_tools.requests.get", return_value=mock_resp):
        res = fetch_corridor_conditions.invoke({"latitude": 33.77, "longitude": -118.19})

    assert "High Transit Disruption" in res
    assert "Extreme ambient heat" in res
    assert "High crosswinds" in res


def test_orchestrator_hitl_pause_and_resume():
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "dispatch_escalation_alert",
                            "args": {
                                "severity": "CRITICAL",
                                "tier": "Tier 2",
                                "coordinates": "33.77, -118.19",
                                "summary": "Fresh perishables at 7.2C",
                                "recommended_action": "Restart cooling & divert to San Bernardino",
                            },
                            "id": "call_hitl_1",
                        }
                    ],
                )
            return AIMessage(content="Escalation ticket dispatched and logged.")

    agent = build_agent(llm=FakeLLM())
    cfg = {"configurable": {"thread_id": "test-hitl-thread"}}

    list(agent.stream({"messages": [HumanMessage(content="Escalate LA breach")]}, config=cfg))
    state = agent.get_state(cfg)
    assert "action_tools" in state.next

    events = list(agent.stream(None, config=cfg, stream_mode="updates"))
    assert any("action_tools" in ev for ev in events)
    final_state = agent.get_state(cfg)
    assert final_state.next == ()
