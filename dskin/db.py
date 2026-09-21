"""SQLite store. Schema is dynamic: metric columns are added as they appear, so
adding a metric never needs a migration."""
from __future__ import annotations
import sqlite3
import os
from . import config as C

_BASE = """
CREATE TABLE IF NOT EXISTS measurements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  captured_at TEXT, ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
  image_id TEXT, path TEXT, arm TEXT, condition TEXT, notes TEXT,
  passed INTEGER, reject_reasons TEXT,
  pipeline_version TEXT,
  UNIQUE(image_id, arm)
);
CREATE INDEX IF NOT EXISTS idx_meas_time ON measurements(captured_at);
"""

# Bump whenever preprocessing changes: embeddings and metrics are NOT comparable
# across versions. Re-ingest the whole history rather than mixing them.
PIPELINE_VERSION = "1.0.0"


def connect(path: str = C.DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(_BASE)
    return con


def _ensure_columns(con, row: dict) -> None:
    have = {r["name"] for r in con.execute("PRAGMA table_info(measurements)")}
    for k, v in row.items():
        if k in have:
            continue
        typ = "REAL" if isinstance(v, (int, float)) and not isinstance(v, bool) else "TEXT"
        con.execute(f'ALTER TABLE measurements ADD COLUMN "{k}" {typ}')


def upsert(con, row: dict) -> None:
    row = {**row, "pipeline_version": PIPELINE_VERSION}
    _ensure_columns(con, row)
    cols = ",".join(f'"{k}"' for k in row)
    ph = ",".join("?" for _ in row)
    con.execute(f"INSERT OR REPLACE INTO measurements ({cols}) VALUES ({ph})",
                list(row.values()))
    con.commit()


def load_df(path: str = C.DB_PATH, arm: str | None = None):
    import pandas as pd
    con = connect(path)
    q = "SELECT * FROM measurements"
    params: list = []
    if arm:
        q += " WHERE arm = ?"
        params.append(arm)
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df
