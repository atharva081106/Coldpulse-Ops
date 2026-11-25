import os
import re
import sqlite3
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LOCAL_DB_PATH = PROJECT_ROOT / "data" / "cache" / "local_fleet.db"
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"

_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|MERGE|EXEC|EXECUTE|GRANT|DENY|REVOKE|INTO)\b",
    re.IGNORECASE,
)

_mssql_ro_engine: Engine | None = None
_mssql_admin_engine: Engine | None = None


def get_backend_mode() -> str:
    """Return active data backend: 'MSSQL' or 'LOCAL'."""
    explicit = os.getenv("DATA_BACKEND", "AUTO").strip().upper()
    if explicit in {"MSSQL", "LOCAL"}:
        return explicit
    if os.getenv("SQL_AGENT_PASSWORD") and os.getenv("SQL_SERVER_HOST"):
        return "MSSQL"
    return "LOCAL"


def validate_readonly_sql(sql_query: str) -> tuple[bool, str]:
    """Validate that a SQL query is a single read-only SELECT/WITH statement."""
    if not sql_query or not sql_query.strip():
        return False, "SECURITY BLOCK: Empty SQL query."

    cleaned = re.sub(r"--[^\n]*", "", sql_query)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL).strip()

    # Allow a single trailing semicolon, reject any internal semicolons
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if ";" in cleaned:
        return False, "SECURITY BLOCK: Multiple SQL statements are not permitted."

    upper = cleaned.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "SECURITY BLOCK: Only SELECT or WITH (CTE) queries are authorized."

    # Strip single-quoted string literals before keyword scanning
    without_strings = re.sub(r"'([^']|'')*'", "''", cleaned)
    match = _FORBIDDEN_SQL_PATTERN.search(without_strings)
    if match:
        return False, f"SECURITY BLOCK: Forbidden SQL keyword '{match.group(1).upper()}' detected."

    return True, cleaned


def _translate_tsql_to_sqlite(sql_query: str) -> str:
    """Translate basic T-SQL idioms (schema prefix, TOP N) to SQLite syntax."""
    translated = re.sub(r"\bFDE_VIEWS\.", "", sql_query, flags=re.IGNORECASE)
    translated = re.sub(r"\bdbo\.", "", translated, flags=re.IGNORECASE)

    top_match = re.match(r"^\s*SELECT\s+TOP\s+(\d+)\s+", translated, flags=re.IGNORECASE)
    if top_match:
        limit_val = top_match.group(1)
        translated = re.sub(
            r"^\s*SELECT\s+TOP\s+\d+\s+",
            "SELECT ",
            translated,
            count=1,
            flags=re.IGNORECASE,
        )
        if not re.search(r"\bLIMIT\s+\d+\b", translated, flags=re.IGNORECASE):
            translated = f"{translated.rstrip()} LIMIT {limit_val}"

    return translated


def _build_mssql_engine(user: str, password: str) -> Engine:
    db_host = os.getenv("SQL_SERVER_HOST", "localhost")
    db_port = os.getenv("SQL_SERVER_PORT", "1433")
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={db_host},{db_port};"
        f"DATABASE=master;"
        f"UID={user};"
        f"PWD={password};"
        f"Encrypt=no;"
        f"TrustServerCertificate=yes;"
    )
    params = urllib.parse.quote_plus(conn_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", pool_pre_ping=True)


def get_mssql_ro_engine() -> Engine:
    global _mssql_ro_engine
    if _mssql_ro_engine is None:
        user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
        password = os.getenv("SQL_AGENT_PASSWORD", "")
        _mssql_ro_engine = _build_mssql_engine(user, password)
    return _mssql_ro_engine


def get_mssql_admin_engine(user: str, password: str) -> Engine:
    global _mssql_admin_engine
    if _mssql_admin_engine is None:
        _mssql_admin_engine = _build_mssql_engine(user, password)
    return _mssql_admin_engine


def ensure_local_db() -> Path:
    """Create and seed the local SQLite database from the raw CSV if needed."""
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_seed = not LOCAL_DB_PATH.exists()

    with sqlite3.connect(LOCAL_DB_PATH) as conn:
        if needs_seed and RAW_CSV_PATH.exists():
            df = pd.read_csv(RAW_CSV_PATH)
            fleet_df = pd.DataFrame(
                {
                    "Timestamp": df["timestamp"],
                    "Latitude": pd.to_numeric(df["vehicle_gps_latitude"], errors="coerce"),
                    "Longitude": pd.to_numeric(df["vehicle_gps_longitude"], errors="coerce"),
                    "Current_Temperature_C": pd.to_numeric(df["iot_temperature"], errors="coerce"),
                    "Cargo_Condition_Code": pd.to_numeric(df["cargo_condition_status"], errors="coerce"),
                    "Risk_Classification": df["risk_classification"].astype(str),
                    "Delay_Probability": pd.to_numeric(df["delay_probability"], errors="coerce"),
                    "Port_Congestion_Level": pd.to_numeric(df["port_congestion_level"], errors="coerce"),
                    "Route_Risk_Index": pd.to_numeric(df["route_risk_level"], errors="coerce"),
                }
            )
            fleet_df.to_sql("VW_ACTIVE_FLEET", conn, if_exists="replace", index=False)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS AgentAuditLog (
                LogID INTEGER PRIMARY KEY AUTOINCREMENT,
                Timestamp TEXT NOT NULL,
                SessionID TEXT,
                NodeExecuted TEXT,
                ToolName TEXT,
                Content TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS EscalationTickets (
                TicketID INTEGER PRIMARY KEY AUTOINCREMENT,
                CreatedAt TEXT NOT NULL,
                SessionID TEXT,
                Severity TEXT,
                Tier TEXT,
                Coordinates TEXT,
                Summary TEXT,
                RecommendedAction TEXT,
                Status TEXT DEFAULT 'DISPATCHED'
            )
            """
        )
        conn.commit()

    return LOCAL_DB_PATH


def execute_readonly_query(sql_query: str, max_rows: int = 25) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Execute a validated read-only query against the active backend."""
    is_valid, result = validate_readonly_sql(sql_query)
    if not is_valid:
        raise ValueError(result)

    mode = get_backend_mode()
    if mode == "MSSQL":
        engine = get_mssql_ro_engine()
        with engine.connect() as conn:
            cursor = conn.execute(text(result))
            columns = list(cursor.keys())
            rows = [tuple(r) for r in cursor.fetchmany(max_rows)]
            return columns, rows

    db_path = ensure_local_db()
    sqlite_sql = _translate_tsql_to_sqlite(result)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(sqlite_sql)
        columns = [desc[0] for desc in (cursor.description or [])]
        rows = cursor.fetchmany(max_rows)
        return columns, rows


def get_fleet_snapshot(limit: int = 300) -> pd.DataFrame:
    """Fetch recent fleet telemetry rows for dashboard KPIs and mapping."""
    mode = get_backend_mode()
    if mode == "MSSQL":
        engine = get_mssql_ro_engine()
        query = f"""
            SELECT TOP {int(limit)}
                Timestamp, Latitude, Longitude, Current_Temperature_C,
                Cargo_Condition_Code, Risk_Classification, Delay_Probability,
                Port_Congestion_Level, Route_Risk_Index
            FROM FDE_VIEWS.VW_ACTIVE_FLEET
            ORDER BY Timestamp DESC
        """
        with engine.connect() as conn:
            return pd.read_sql(query, conn)

    db_path = ensure_local_db()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            f"""
            SELECT Timestamp, Latitude, Longitude, Current_Temperature_C,
                   Cargo_Condition_Code, Risk_Classification, Delay_Probability,
                   Port_Congestion_Level, Route_Risk_Index
            FROM VW_ACTIVE_FLEET
            ORDER BY Timestamp DESC
            LIMIT {int(limit)}
            """,
            conn,
        )


def write_audit_log(session_id: str, node_name: str, tool_name: str, content: str) -> None:
    """Persist an agent execution trace to AgentAuditLog."""
    mode = get_backend_mode()
    try:
        if mode == "MSSQL":
            engine = get_mssql_ro_engine()
            with engine.connect() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO FDE_VIEWS.AgentAuditLog (SessionID, NodeExecuted, ToolName, Content)
                        VALUES (:session_id, :node_name, :tool_name, :content)
                        """
                    ),
                    {
                        "session_id": session_id,
                        "node_name": node_name,
                        "tool_name": tool_name,
                        "content": content,
                    },
                )
                conn.commit()
            return

        db_path = ensure_local_db()
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO AgentAuditLog (Timestamp, SessionID, NodeExecuted, ToolName, Content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now_iso, session_id, node_name, tool_name, content),
            )
            conn.commit()
    except Exception as exc:
        print(f"Audit Log Failed (Silent): {exc}")


def read_audit_logs(admin_user: str = "", admin_pass: str = "") -> pd.DataFrame:
    """Read audit logs from the active backend."""
    mode = get_backend_mode()
    if mode == "MSSQL":
        engine = get_mssql_admin_engine(admin_user, admin_pass)
        with engine.connect() as conn:
            return pd.read_sql(
                """
                SELECT LogID, Timestamp, SessionID, NodeExecuted, ToolName, Content
                FROM FDE_VIEWS.AgentAuditLog
                ORDER BY Timestamp DESC
                """,
                conn,
            )

    db_path = ensure_local_db()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            """
            SELECT LogID, Timestamp, SessionID, NodeExecuted, ToolName, Content
            FROM AgentAuditLog
            ORDER BY LogID DESC
            """,
            conn,
        )


def create_escalation_ticket(
    session_id: str,
    severity: str,
    tier: str,
    coordinates: str,
    summary: str,
    recommended_action: str,
) -> int:
    """Create a dispatch escalation ticket in the local/enterprise store."""
    db_path = ensure_local_db()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO EscalationTickets
                (CreatedAt, SessionID, Severity, Tier, Coordinates, Summary, RecommendedAction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now_iso, session_id, severity, tier, coordinates, summary, recommended_action),
        )
        conn.commit()
        return int(cur.lastrowid)


def read_escalation_tickets(limit: int = 50) -> pd.DataFrame:
    """Fetch recent dispatch escalation tickets."""
    db_path = ensure_local_db()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            f"""
            SELECT TicketID, CreatedAt, Severity, Tier, Coordinates, Summary, RecommendedAction, Status
            FROM EscalationTickets
            ORDER BY TicketID DESC
            LIMIT {int(limit)}
            """,
            conn,
        )

