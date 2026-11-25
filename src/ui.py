import json
import os
import sys
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

# ==========================================
# 1. PATH & ENVIRONMENT RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
POLICY_DIR = project_root / "data" / "policy"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.db_backend import (
    get_backend_mode,
    get_fleet_snapshot,
    read_audit_logs,
    read_escalation_tickets,
    write_audit_log,
)
from src.orchestrator import get_fde_agent

NVIDIA_MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
]

# ==========================================
# 2. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="CryoGuard AI • Cold-Chain Dispatch Console",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; }
    div[data-testid="stSidebar"] { background-color: #111622; border-right: 1px solid #1E293B; }
    .stMarkdown code { background-color: #1E293B !important; color: #38BDF8 !important; }
    div[data-testid="stMetric"] {
        background-color: #111622;
        border: 1px solid #1E293B;
        padding: 12px 16px;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. SESSION STATE
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = os.getenv(
        "NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
    )

thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}


def active_model_override() -> str | None:
    if os.getenv("Agent_llm", "OLLAMA").strip().upper() == "NVIDIA":
        return st.session_state.selected_model
    return None


@st.cache_data(ttl=60, show_spinner=False)
def load_cached_fleet() -> pd.DataFrame:
    try:
        return get_fleet_snapshot(limit=300)
    except Exception:
        return pd.DataFrame()


def run_agent_stream(stream_input) -> tuple[str, list[dict]]:
    """Execute or resume the LangGraph stream with live token streaming."""
    final_response = ""
    streamed_text = ""
    current_traces: list[dict] = []

    agent = get_fde_agent(model_override=active_model_override())

    with st.status("🧠 Initializing Core Reasoner Node...", expanded=True) as status:
        response_placeholder = st.empty()
        events = agent.stream(
            stream_input,
            config=thread_config,
            stream_mode=["updates", "messages"],
        )

        for mode, payload in events:
            if mode == "messages":
                msg_chunk, metadata = payload
                if (
                    metadata.get("langgraph_node") == "reasoner"
                    and isinstance(msg_chunk, AIMessageChunk)
                    and isinstance(msg_chunk.content, str)
                    and msg_chunk.content
                    and not msg_chunk.tool_call_chunks
                ):
                    streamed_text += msg_chunk.content
                    status.update(label="📝 Streaming Operational Resolution Report...")
                    response_placeholder.markdown(streamed_text + "▌")
                continue

            for node_name, node_state in payload.items():
                if node_name == "reasoner":
                    latest_msg = node_state["messages"][-1]
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        status.update(label="🧠 Agent generated tool parameters...")
                        for tool_call in latest_msg.tool_calls:
                            st.markdown(f"**⚡ Intent Recognized:** `{tool_call['name']}`")
                            with st.expander(f"📥 View Generated Input ({tool_call['name']})", expanded=False):
                                st.json(tool_call["args"])

                            current_traces.append(
                                {
                                    "type": "tool_input",
                                    "name": tool_call["name"],
                                    "args": tool_call["args"],
                                }
                            )
                            write_audit_log(
                                session_id=st.session_state.thread_id,
                                node_name="reasoner",
                                tool_name=tool_call["name"],
                                content=json.dumps(tool_call["args"]),
                            )

                    if latest_msg.content:
                        final_response = latest_msg.content
                        write_audit_log(
                            session_id=st.session_state.thread_id,
                            node_name="reasoner_final",
                            tool_name="LLM Text Synthesis",
                            content=final_response,
                        )

                elif node_name in {"tools", "action_tools"}:
                    status.update(label="🔧 Executing Enterprise Subsystem Tools...")
                    for msg in node_state.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            with st.expander(f"📤 View Raw Output ({msg.name})", expanded=False):
                                st.code(msg.content, language="text")

                            current_traces.append(
                                {
                                    "type": "tool_output",
                                    "name": msg.name,
                                    "content": msg.content,
                                }
                            )
                            write_audit_log(
                                session_id=st.session_state.thread_id,
                                node_name=node_name,
                                tool_name=msg.name,
                                content=msg.content,
                            )

        response_placeholder.empty()
        status.update(label="Incident Matrix Evaluation Complete", state="complete", expanded=False)

    return final_response or streamed_text, current_traces


def render_hitl_gate() -> None:
    """Render approval controls if the graph is paused before action_tools."""
    try:
        agent = get_fde_agent(model_override=active_model_override())
        snapshot = agent.get_state(thread_config)
    except Exception:
        return
    if not snapshot or "action_tools" not in (snapshot.next or ()):
        return

    last_msg = snapshot.values["messages"][-1]
    pending_calls = getattr(last_msg, "tool_calls", [])
    if not pending_calls:
        return

    st.warning("⚠️ **Human-in-the-Loop Approval Required** — Agent requested an escalation dispatch:")
    for call in pending_calls:
        st.json(call["args"])

    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("✅ Approve & Dispatch Escalation", type="primary", width="stretch"):
            with st.chat_message("assistant", avatar="🤖"):
                final_resp, traces = run_agent_stream(None)
                if final_resp:
                    st.markdown(final_resp)
                    st.session_state.ui_messages.append(
                        {"role": "assistant", "content": final_resp, "traces": traces}
                    )
            st.rerun()

    with col_reject:
        if st.button("❌ Reject Escalation", width="stretch"):
            rejection_msgs = [
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    content="DISPATCHER REJECTED: Escalation cancelled by human operator.",
                )
                for call in pending_calls
            ]
            agent.update_state(thread_config, {"messages": rejection_msgs}, as_node="action_tools")
            with st.chat_message("assistant", avatar="🤖"):
                final_resp, traces = run_agent_stream(None)
                if final_resp:
                    st.markdown(final_resp)
                    st.session_state.ui_messages.append(
                        {"role": "assistant", "content": final_resp, "traces": traces}
                    )
            st.rerun()


def render_fleet_analytics(fleet_df: pd.DataFrame) -> None:
    """Render PyDeck map, Altair charts, and clickable high-risk shipment triage."""
    if fleet_df.empty:
        st.info("Fleet telemetry unavailable.")
        return

    tab_map, tab_charts, tab_triage = st.tabs(
        ["🗺️ Live PyDeck Map", "📈 Telemetry Analytics", "🎯 High-Risk Triage"]
    )

    with tab_map:
        map_df = fleet_df.dropna(subset=["Latitude", "Longitude"]).copy()
        map_df["Temp_C"] = map_df["Current_Temperature_C"].round(2)
        map_df["Delay_Prob"] = map_df["Delay_Probability"].round(2)
        map_df["Port_Congestion"] = map_df["Port_Congestion_Level"].round(2)
        map_df["r"] = map_df["Current_Temperature_C"].apply(lambda t: 239 if t > 4.0 else 56)
        map_df["g"] = map_df["Current_Temperature_C"].apply(lambda t: 68 if t > 4.0 else 189)
        map_df["b"] = map_df["Current_Temperature_C"].apply(lambda t: 68 if t > 4.0 else 248)

        view_state = pdk.ViewState(
            latitude=float(map_df["Latitude"].mean()),
            longitude=float(map_df["Longitude"].mean()),
            zoom=3.3,
            pitch=20,
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[Longitude, Latitude]",
            get_fill_color="[r, g, b, 195]",
            get_radius=32000,
            pickable=True,
        )
        tooltip = {
            "html": (
                "<b>Risk:</b> {Risk_Classification}<br/>"
                "<b>Temp:</b> {Temp_C} °C<br/>"
                "<b>Delay Prob:</b> {Delay_Prob}<br/>"
                "<b>Port Congestion:</b> {Port_Congestion}"
            ),
            "style": {"backgroundColor": "#111622", "color": "#E2E8F0"},
        }
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            hist = (
                alt.Chart(fleet_df)
                .mark_bar(color="#38BDF8")
                .encode(
                    x=alt.X("Current_Temperature_C:Q", bin=alt.Bin(maxbins=24), title="IoT Temp (°C)"),
                    y=alt.Y("count()", title="Shipments"),
                )
                .properties(title="Temperature Distribution vs 4.0°C SOP Limit", height=250)
            )
            rule = alt.Chart(pd.DataFrame({"limit": [4.0]})).mark_rule(color="#EF4444", strokeWidth=2).encode(
                x="limit:Q"
            )
            st.altair_chart(hist + rule, width="stretch")

        with c2:
            scatter = (
                alt.Chart(fleet_df)
                .mark_circle(size=55, opacity=0.75)
                .encode(
                    x=alt.X("Route_Risk_Index:Q", title="Route Risk Index"),
                    y=alt.Y("Delay_Probability:Q", title="Delay Probability"),
                    color=alt.Color("Risk_Classification:N", legend=alt.Legend(title="Risk")),
                    tooltip=["Current_Temperature_C", "Port_Congestion_Level", "Risk_Classification"],
                )
                .properties(title="Delay Probability vs Route Risk", height=250)
            )
            st.altair_chart(scatter, width="stretch")

    with tab_triage:
        breach_df = (
            fleet_df[fleet_df["Current_Temperature_C"] > 4.0]
            .sort_values("Current_Temperature_C", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        if breach_df.empty:
            st.success("No temperature excursions above 4.0°C in the current sample.")
        else:
            event = st.dataframe(
                breach_df[
                    [
                        "Timestamp",
                        "Latitude",
                        "Longitude",
                        "Current_Temperature_C",
                        "Risk_Classification",
                        "Delay_Probability",
                        "Port_Congestion_Level",
                    ]
                ],
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
            )
            selected_rows = event.selection.get("rows", []) if event and event.selection else []
            if selected_rows:
                row = breach_df.iloc[selected_rows[0]]
                if st.button(
                    f"🔍 Investigate Selected Shipment ({row['Latitude']:.2f}, {row['Longitude']:.2f})",
                    type="primary",
                    width="stretch",
                ):
                    st.session_state.pending_prompt = (
                        f"Investigate the shipment at Latitude {row['Latitude']:.4f}, "
                        f"Longitude {row['Longitude']:.4f} reporting {row['Current_Temperature_C']:.2f}°C "
                        f"(Risk: {row['Risk_Classification']}, Delay Prob: {row['Delay_Probability']:.2f}, "
                        f"Port Congestion: {row['Port_Congestion_Level']:.2f}). Check corridor weather, "
                        f"cite the SOP rules, and recommend an action plan."
                    )
                    st.rerun()


# ==========================================
# 4. SIDEBAR NAVIGATION & METADATA
# ==========================================
backend_mode = get_backend_mode()
llm_provider = os.getenv("Agent_llm", "OLLAMA").strip().upper()

with st.sidebar:
    st.image(
        str(script_dir / "image_L25X5q.png")
        if (script_dir / "image_L25X5q.png").exists()
        else "https://cdn-icons-png.flaticon.com/512/2830/2830305.png",
        width=65,
    )
    st.title("CryoGuard Command")

    app_mode = st.radio(
        "System Mode",
        [
            "🧊 Dispatch Console",
            "🚨 Escalation Tickets",
            "📜 SOP Policy Manager",
            "🛡️ Security & Audit Logs",
        ],
    )

    st.markdown("---")
    st.caption(f"Session Token: `{st.session_state.thread_id[:8]}...`")
    st.markdown(f"**Data Backend:** `{backend_mode}`")
    st.markdown(f"**Reasoning Provider:** `{llm_provider}`")

    if llm_provider == "NVIDIA":
        st.session_state.selected_model = st.selectbox(
            "NVIDIA NIM Model",
            options=NVIDIA_MODELS,
            index=NVIDIA_MODELS.index(st.session_state.selected_model)
            if st.session_state.selected_model in NVIDIA_MODELS
            else 0,
        )

    st.markdown(f"**SOP Store:** `{'PINECONE' if os.getenv('PINECONE_API_KEY') else 'LOCAL_SOP'}`")

    st.markdown("---")
    if st.button("🗑️ Purge Dispatch Session", width="stretch"):
        st.session_state.ui_messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_prompt = None
        st.rerun()

# ==========================================
# 5. VIEW ROUTING
# ==========================================
fleet_df = load_cached_fleet()
tickets_df = read_escalation_tickets()

if app_mode == "🧊 Dispatch Console":
    st.title("Cold-Chain Incident Control Dashboard")
    st.caption("Real-Time Fleet Telemetry • Corridor Weather Intelligence • HITL SOP Dispatch")

    k1, k2, k3, k4 = st.columns(4)
    if not fleet_df.empty:
        breaches = int((fleet_df["Current_Temperature_C"] > 4.0).sum())
        high_risk = int((fleet_df["Risk_Classification"].astype(str).str.contains("High", case=False)).sum())
        k1.metric("Tracked Shipments (Sample)", len(fleet_df))
        k2.metric("Temp Excursions (>4.0°C)", breaches)
        k3.metric("High-Risk Classifications", high_risk)
    else:
        k1.metric("Tracked Shipments", "N/A")
        k2.metric("Temp Excursions (>4.0°C)", "N/A")
        k3.metric("High-Risk Classifications", "N/A")
    k4.metric("Escalation Tickets", len(tickets_df))

    with st.expander("📊 Live Fleet Map, Analytics & Anomaly Triage", expanded=False):
        render_fleet_analytics(fleet_df)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Scan Active Temp Breaches (>4.0°C)", width="stretch"):
                st.session_state.pending_prompt = (
                    "Query FDE_VIEWS.VW_ACTIVE_FLEET for the top 3 shipments with "
                    "Current_Temperature_C > 4.0 and High Risk classification, check corridor weather "
                    "for the worst one, and recommend SOP actions."
                )
        with c2:
            if st.button("⚓ Check Port Congestion & Diversions", width="stretch"):
                st.session_state.pending_prompt = (
                    "Find active shipments with Port_Congestion_Level > 7.0, verify SOP diversion "
                    "rules, and provide an executive action plan."
                )

    for entry in st.session_state.ui_messages:
        with st.chat_message(entry["role"], avatar="👤" if entry["role"] == "user" else "🤖"):
            for trace in entry.get("traces", []):
                if trace["type"] == "tool_input":
                    st.markdown(f"**⚡ Intent Recognized:** `{trace['name']}`")
                    with st.expander(f"📥 View Generated Input ({trace['name']})", expanded=False):
                        st.json(trace["args"])
                elif trace["type"] == "tool_output":
                    with st.expander(f"📤 View Raw Output ({trace['name']})", expanded=False):
                        st.code(trace["content"], language="text")
            st.markdown(entry["content"])

    render_hitl_gate()

    chat_input = st.chat_input("Query fleet telemetry, corridor weather, or SOP thresholds...")
    user_input = st.session_state.pending_prompt or chat_input
    if st.session_state.pending_prompt:
        st.session_state.pending_prompt = None

    if user_input:
        st.session_state.ui_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            final_response, current_traces = run_agent_stream({"messages": [HumanMessage(content=user_input)]})
            if final_response:
                st.markdown(final_response)
                st.session_state.ui_messages.append(
                    {
                        "role": "assistant",
                        "content": final_response,
                        "traces": current_traces,
                    }
                )
            else:
                try:
                    snapshot = get_fde_agent(model_override=active_model_override()).get_state(thread_config)
                except Exception:
                    snapshot = None
                if snapshot and "action_tools" in (snapshot.next or ()):
                    st.info("⏸️ Execution paused for dispatcher approval below.")
                    st.rerun()
                else:
                    st.error("⚠️ Execution completed without a final text response.")

elif app_mode == "🚨 Escalation Tickets":
    st.title("🚨 Dispatch Escalation Tickets")
    st.caption("Human-approved cold-chain breach & Tier 2 escalation log")

    if not tickets_df.empty:
        st.dataframe(tickets_df, hide_index=True, width="stretch", height=550)
    else:
        st.info("No escalation tickets logged yet.")

elif app_mode == "📜 SOP Policy Manager":
    st.title("📜 Standard Operating Procedures (SOPs)")
    st.caption("Inspect active compliance rules or upload new Markdown/Text SOP documents")

    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = st.file_uploader("Upload New SOP Policy (.md or .txt)", type=["md", "txt"])
    if uploaded is not None:
        target_name = Path(uploaded.name).stem + ".md"
        target_path = POLICY_DIR / target_name
        target_path.write_bytes(uploaded.getvalue())
        st.success(f"✅ Saved `{target_name}` to `data/policy/`. Local SOP retriever will index it automatically.")

    sop_files = sorted(POLICY_DIR.glob("*.md"))
    if sop_files:
        selected_sop = st.selectbox("Active Policy Documents", [f.name for f in sop_files])
        st.markdown((POLICY_DIR / selected_sop).read_text(encoding="utf-8"))
    else:
        st.info("No SOP policy files found in data/policy/.")

elif app_mode == "🛡️ Security & Audit Logs":
    st.title("🛡️ Enterprise Agent Audit Trail")
    st.caption(f"Inspection of AgentAuditLog ({backend_mode} backend)")

    if backend_mode == "LOCAL":
        df = read_audit_logs()
        if not df.empty:
            st.dataframe(df, hide_index=True, width="stretch", height=600)
        else:
            st.info("No audit logs found yet. Run a query in the Dispatch Console first.")
    else:
        st.markdown("### Database Authorization Gate")
        st.markdown("Enter `SQL_ADMIN_USER` credentials to query `FDE_VIEWS.AgentAuditLog`.")
        with st.form("admin_auth_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_user = st.text_input("Admin Username", value=os.getenv("SQL_ADMIN_USER", ""))
            with col2:
                input_pass = st.text_input("Admin Password", type="password", value="")
            submit_admin = st.form_submit_button("Authenticate & Load Logs", width="stretch")

        if submit_admin:
            expected_admin_user = os.getenv("SQL_ADMIN_USER")
            expected_admin_pass = os.getenv("SQL_ADMIN_PASSWORD")
            if input_user == expected_admin_user and input_pass == expected_admin_pass:
                try:
                    df = read_audit_logs(input_user, input_pass)
                    st.success("✅ Authenticated successfully as Admin.")
                    if not df.empty:
                        st.dataframe(df, hide_index=True, width="stretch", height=600)
                    else:
                        st.info("No audit logs found in the database.")
                except Exception as exc:
                    st.error(f"Database Query Failed: {exc}")
            else:
                st.error("❌ Invalid Administrator Credentials.")
