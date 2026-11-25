import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from src.db_backend import create_escalation_ticket, execute_readonly_query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

POLICY_DIR = PROJECT_ROOT / "data" / "policy"

_retriever: Any = None


def _get_pinecone_retriever():
    """Lazily initialize Pinecone retriever only when configured."""
    global _retriever
    if _retriever is not None:
        return _retriever

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    sop_backend = os.getenv("SOP_BACKEND", "AUTO").strip().upper()
    if sop_backend == "LOCAL" or not pinecone_api_key:
        return None

    from langchain_pinecone import PineconeVectorStore

    embeddings_setting = os.getenv("Embeddings_model", "LOCAL").strip().upper()
    if embeddings_setting == "OPENAI":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        index_name = "fde-sop-index-openai"
    else:
        local_model_target = os.getenv("Local_Embedding_Model", "BAAI/bge-m3").strip()
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=local_model_target,
            model_kwargs={"device": "cpu"},
        )
        index_name = "fde-sop-index-local"

    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    _retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return _retriever


def _search_local_sop(query: str, top_k: int = 3) -> str:
    """Search local Markdown SOP files by section using TF-IDF or keyword overlap."""
    if not POLICY_DIR.exists():
        return "No local SOP policy directory found."

    sections: list[tuple[str, str]] = []
    for md_file in sorted(POLICY_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        # Split on level-2/3 markdown headings to preserve rule context
        chunks = [c.strip() for c in content.split("\n### ") if c.strip()]
        for idx, chunk in enumerate(chunks):
            text_block = chunk if idx == 0 else f"### {chunk}"
            sections.append((md_file.name, text_block))

    if not sections:
        return "No SOP documents found in data/policy."

    corpus = [sec[1] for sec in sections]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus + [query])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]
        selected = [(sections[i][0], sections[i][1], float(sims[i])) for i in ranked_idx]
    except Exception:
        query_terms = set(query.lower().split())
        scored = []
        for fname, block in sections:
            words = set(block.lower().split())
            scored.append((fname, block, float(len(query_terms & words))))
        scored.sort(key=lambda item: item[2], reverse=True)
        selected = scored[:top_k]

    formatted = "\n\n".join(
        f"[Source: {fname} | Mode: LOCAL_SOP | Relevance: {score:.2f}]\n{block}"
        for fname, block, score in selected
    )
    return f"--- COMPLIANCE SOP CONTEXT ---\n{formatted}\n------------------------------"


def _format_markdown_table(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Format SQL query results as a clean Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in rows:
        cells = [
            f"{val:.4f}" if isinstance(val, float) else str(val)
            for val in row
        ]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body_lines])


@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a read-only SQL SELECT query against the FDE_VIEWS.VW_ACTIVE_FLEET view.
    Columns available:
    Timestamp, Latitude, Longitude, Current_Temperature_C, Cargo_Condition_Code,
    Risk_Classification, Delay_Probability, Port_Congestion_Level, Route_Risk_Index.
    Always write standard T-SQL queries (single SELECT or WITH statement).
    """
    try:
        columns, rows = execute_readonly_query(sql_query, max_rows=20)
        if not rows:
            return "No records matched the query criteria."
        table = _format_markdown_table(columns, rows)
        return f"Returned {len(rows)} row(s):\n\n{table}"
    except ValueError as sec_err:
        return str(sec_err)
    except Exception as exc:
        return f"Database Error: {exc}"


@tool
def fetch_corridor_conditions(latitude: float, longitude: float) -> str:
    """
    Fetches real-time weather, ambient temperature stress, and 6-hour forecast hazards
    from Open-Meteo for given GPS coordinates.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&current_weather=true"
            f"&hourly=temperature_2m,precipitation_probability,weathercode"
            f"&forecast_days=1"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()

        current = data.get("current_weather", {})
        temp = current.get("temperature")
        wind = float(current.get("windspeed", 0.0) or 0.0)
        weather_code = int(current.get("weathercode", 0) or 0)

        hourly = data.get("hourly", {})
        precip_probs = hourly.get("precipitation_probability", [])[:6]
        max_precip_prob = max(precip_probs) if precip_probs else 0

        # Compute multi-factor corridor risk score (0-10)
        risk_score = 1.5
        risk_factors: list[str] = []

        if isinstance(temp, (int, float)):
            if temp >= 32.0:
                risk_score += 3.5
                risk_factors.append(f"Extreme ambient heat ({temp}°C) straining reefer cooling")
            elif temp >= 25.0:
                risk_score += 1.5
                risk_factors.append(f"Elevated ambient heat ({temp}°C)")
            elif temp <= -10.0:
                risk_score += 2.5
                risk_factors.append(f"Severe freeze conditions ({temp}°C)")

        if wind >= 35.0:
            risk_score += 3.0
            risk_factors.append(f"High crosswinds ({wind} km/h)")
        elif wind >= 18.0:
            risk_score += 1.5
            risk_factors.append(f"Moderate wind ({wind} km/h)")

        if max_precip_prob >= 60:
            risk_score += 2.5
            risk_factors.append(f"High precipitation probability ({max_precip_prob}% next 6h)")

        if weather_code >= 61:
            risk_score += 2.0
            risk_factors.append(f"Adverse weather code ({weather_code})")

        risk_score = min(10.0, round(risk_score, 1))
        if risk_score >= 7.0:
            status_note = "High Transit Disruption"
        elif risk_score >= 4.0:
            status_note = "Moderate Weather Caution"
        else:
            status_note = "Corridor Normal"

        factors_text = "; ".join(risk_factors) if risk_factors else "No significant weather hazards"

        return (
            f"--- LIVE CORRIDOR TELEMETRY ---\n"
            f"Target GPS: {latitude}, {longitude}\n"
            f"External Temp: {temp}°C | Wind Speed: {wind} km/h | 6h Max Precip Prob: {max_precip_prob}%\n"
            f"Corridor Risk: {status_note} (Risk Score: {risk_score}/10)\n"
            f"Hazard Factors: {factors_text}\n"
            f"-------------------------------"
        )
    except Exception as exc:
        return f"Corridor API Communication Failure: {exc}"


@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches enterprise Standard Operating Procedures (SOPs) for regulatory thresholds,
    cold-chain breach mitigations, and rerouting/escalation rules.
    Uses Pinecone when configured, with automatic local SOP fallback.
    """
    try:
        retriever = _get_pinecone_retriever()
        if retriever is None:
            return _search_local_sop(query)

        matched_docs = retriever.invoke(query)
        if not matched_docs:
            return _search_local_sop(query)

        formatted_context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source_file', 'SOP')} | Format: {doc.metadata.get('file_format', 'RAW')}]\n{doc.page_content}"
                for doc in matched_docs
            ]
        )
        return f"--- COMPLIANCE SOP CONTEXT ---\n{formatted_context}\n------------------------------"
    except Exception as exc:
        return f"{_search_local_sop(query)}\n\n(Pinecone fallback triggered: {exc})"


@tool
def dispatch_escalation_alert(
    severity: str,
    tier: str,
    coordinates: str,
    summary: str,
    recommended_action: str,
) -> str:
    """
    Logs a formal cold-chain escalation ticket and dispatches an operational alert.
    Use ONLY when an active SOP breach, port diversion, or Tier 2 escalation is confirmed
    and requires immediate operational dispatch tracking.
    Args:
        severity: e.g., 'CRITICAL', 'HIGH', 'MODERATE'
        tier: e.g., 'Tier 1', 'Tier 2'
        coordinates: Shipment GPS coordinates, e.g., '33.77, -118.19'
        summary: Concise description of the anomaly and SOP violation
        recommended_action: Specific mitigation step from the SOP
    """
    ticket_id = create_escalation_ticket(
        session_id="agent-dispatch",
        severity=severity.upper(),
        tier=tier,
        coordinates=coordinates,
        summary=summary,
        recommended_action=recommended_action,
    )

    webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    webhook_status = "Local ticket logged (no webhook configured)"
    if webhook_url:
        try:
            resp = requests.post(
                webhook_url,
                json={
                    "ticket_id": ticket_id,
                    "severity": severity.upper(),
                    "tier": tier,
                    "coordinates": coordinates,
                    "summary": summary,
                    "recommended_action": recommended_action,
                },
                timeout=5,
            )
            resp.raise_for_status()
            webhook_status = f"Webhook notified (HTTP {resp.status_code})"
        except Exception as exc:
            webhook_status = f"Ticket logged, webhook failed: {exc}"

    return (
        f"ESCALATION DISPATCHED | Ticket #{ticket_id} | Severity: {severity.upper()} | "
        f"{tier} | GPS: {coordinates} | Delivery: {webhook_status}"
    )


if __name__ == "__main__":
    print("\n--- Testing Tool 1: Telemetry View ---")
    print(query_telemetry_db.invoke("SELECT TOP 2 Latitude, Longitude, Current_Temperature_C FROM FDE_VIEWS.VW_ACTIVE_FLEET"))

    print("\n--- Testing Tool 2: Live Corridor API ---")
    print(fetch_corridor_conditions.invoke({"latitude": 33.77, "longitude": -118.19}))

    print("\n--- Testing Tool 3: Compliance SOP Retrieval ---")
    print(search_compliance_sop.invoke("What are the temperature rules for fresh perishables?"))
