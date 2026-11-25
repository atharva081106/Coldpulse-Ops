import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. SETUP & PATH RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_tools import (
    dispatch_escalation_alert,
    fetch_corridor_conditions,
    query_telemetry_db,
    search_compliance_sop,
)

load_dotenv(project_root / ".env")

PROMPT_PATH = project_root / "src" / "prompts" / "system_prompt.txt"
CHECKPOINT_DB_PATH = project_root / "data" / "cache" / "langgraph_checkpoints.db"

READONLY_TOOLS = [query_telemetry_db, fetch_corridor_conditions, search_compliance_sop]
ACTION_TOOLS = [dispatch_escalation_alert]
ALL_TOOLS = READONLY_TOOLS + ACTION_TOOLS
ACTION_TOOL_NAMES = {t.name for t in ACTION_TOOLS}


def load_system_prompt() -> str:
    """Load the structured operational system prompt from disk."""
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "You are a helpful cold-chain logistics AI assistant."


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 2. FACTORY INITIALIZATION: AGENT REASONER LLM
# ==========================================
def build_llm(model_override: str | None = None):
    """Initialize the configured chat model from environment variables."""
    agent_llm_setting = os.getenv("Agent_llm", "OLLAMA").strip().upper()

    if agent_llm_setting == "OPENAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_override or os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

    if agent_llm_setting == "NVIDIA":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_override or os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            temperature=0,
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            max_tokens=2048,
        )

    if agent_llm_setting == "DEEPSEEK":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_override or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            temperature=0,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            max_tokens=2048,
        )

    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_override or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=0,
            num_predict=1024,
        )
    except ImportError:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_override or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            temperature=0,
        )


def _build_checkpointer():
    """Use SQLite checkpointing if available, falling back to MemorySaver."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        return MemorySaver()


def route_tools(state: AgentState) -> str:
    """Route tool calls to read-only tools, HITL action tools, or END."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
        return END

    called_names = {tc["name"] for tc in last_msg.tool_calls}
    if called_names & ACTION_TOOL_NAMES:
        return "action_tools"
    return "tools"


def build_agent(llm=None):
    """Compile the LangGraph FDE orchestrator with HITL gate on escalation actions."""
    model = llm or build_llm()
    llm_with_tools = model.bind_tools(ALL_TOOLS)
    system_instructions = load_system_prompt()

    def reasoning_node(state: AgentState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_instructions), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("reasoner", reasoning_node)
    graph_builder.add_node("tools", ToolNode(READONLY_TOOLS))
    graph_builder.add_node("action_tools", ToolNode(ACTION_TOOLS))

    graph_builder.add_edge(START, "reasoner")
    graph_builder.add_conditional_edges(
        "reasoner",
        route_tools,
        {
            "tools": "tools",
            "action_tools": "action_tools",
            END: END,
        },
    )
    graph_builder.add_edge("tools", "reasoner")
    graph_builder.add_edge("action_tools", "reasoner")

    return graph_builder.compile(
        checkpointer=_build_checkpointer(),
        interrupt_before=["action_tools"],
    )


_fde_agents: dict[str, object] = {}


def get_fde_agent(model_override: str | None = None):
    """Lazily build and cache the compiled LangGraph orchestrator per model."""
    cache_key = model_override or "__default__"
    if cache_key not in _fde_agents:
        llm = build_llm(model_override=model_override)
        _fde_agents[cache_key] = build_agent(llm=llm)
    return _fde_agents[cache_key]


# ==========================================
# 3. CLI TESTING PANEL
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("🚀 FDE Supply Chain Orchestrator State Machine Online")
    print(f"   LLM: {os.getenv('Agent_llm', 'OLLAMA')} | SOP: {os.getenv('SOP_BACKEND', 'AUTO')}")
    print("=" * 55 + "\n")

    thread_config = {"configurable": {"thread_id": "production_test_1"}}

    agent = get_fde_agent()
    while True:
        user_input = input("\nDispatcher > ")
        if user_input.lower() in ["exit", "quit"]:
            break

        events = agent.stream(
            {"messages": [("user", user_input)]},
            config=thread_config,
            stream_mode="updates",
        )
        for event in events:
            for node_name, node_state in event.items():
                if node_name == "tools":
                    print("   [System] 🔄 Retrieved telemetry/SOP data...")
                elif node_name == "reasoner":
                    latest_msg = node_state["messages"][-1]
                    if latest_msg.content:
                        print(f"\n🤖 FDE Agent:\n{latest_msg.content}")
