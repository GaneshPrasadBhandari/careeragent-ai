from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from careeragent.config import artifacts_root


class SqliteStateStore:
    """
    Description: Local-first persistence for OrchestrationState using SQLite.
    Layer: L8
    Input: state JSON snapshots + action logs
    Output: durable run state + polling
    """

    def __init__(self) -> None:
        """
        Description: Initialize sqlite DB under artifacts/db/.
        Layer: L0
        Input: None
        Output: SqliteStateStore
        """
        db_dir = artifacts_root() / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / "careeragent.db"
        self._init_schema()

    def _init_schema(self) -> None:
        """
        Description: Create tables if they do not exist.
        Layer: L0
        Input: None
        Output: SQLite schema initialized
        """
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            con.commit()

    def upsert_state(self, *, run_id: str, status: str, state: Dict[str, Any], updated_at_utc: str) -> None:
        """
        Description: Upsert run state snapshot.
        Layer: L8
        Input: run_id + status + state JSON + timestamp
        Output: persisted snapshot
        """
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                """
                INSERT INTO runs(run_id, status, state_json, updated_at_utc)
                VALUES(?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    state_json=excluded.state_json,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (run_id, status, json.dumps(state), updated_at_utc),
            )
            con.commit()

    def get_state(self, *, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Description: Load latest run state snapshot.
        Layer: L8
        Input: run_id
        Output: state dict or None
        """
        with sqlite3.connect(self._db_path) as con:
            cur = con.execute("SELECT state_json FROM runs WHERE run_id=?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])

    def insert_action(self, *, run_id: str, action_type: str, payload: Dict[str, Any], created_at_utc: str) -> None:
        """
        Description: Store HITL actions for audit.
        Layer: L5
        Input: action record
        Output: persisted action row
        """
        with sqlite3.connect(self._db_path) as con:
            con.execute(
                "INSERT INTO actions(run_id, action_type, payload_json, created_at_utc) VALUES(?,?,?,?)",
                (run_id, action_type, json.dumps(payload), created_at_utc),
            )
            con.commit()
