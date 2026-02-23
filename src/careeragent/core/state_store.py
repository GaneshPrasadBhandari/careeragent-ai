from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from careeragent.core.mcp_client import MCPClient, sqlite_path_from_database_url
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState


class StateStore:
    """Description: Persist and load AgentState.
    Layer: L8
    Input: AgentState
    Output: saved state snapshot
    """

    def __init__(self, settings: Settings, mcp: MCPClient) -> None:
        self.s = settings
        self.mcp = mcp
        self._locks: Dict[str, threading.Lock] = {}

    def _lock_for(self, run_id: str) -> threading.Lock:
        if run_id not in self._locks:
            self._locks[run_id] = threading.Lock()
        return self._locks[run_id]

    def run_dir(self, run_id: str) -> Path:
        return Path("outputs/runs") / run_id

    def save(self, state: AgentState) -> None:
        """Save state snapshot to file + sqlite."""
        run_id = state.run_id
        with self._lock_for(run_id):
            rd = self.run_dir(run_id)
            rd.mkdir(parents=True, exist_ok=True)
            path = rd / "state.json"
            payload = json.dumps(state.model_dump(mode="json"), indent=2).encode("utf-8")
            self.mcp.write_file(str(path), payload)

            # sqlite
            db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
            self._ensure_tables(db_path)
            self.mcp.sqlite_exec(
                db_path,
                "INSERT OR REPLACE INTO runs(run_id, status, pending_action, progress_percent, current_layer, state_json) VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    state.status,
                    state.pending_action,
                    int(state.progress_percent),
                    state.current_layer,
                    payload.decode("utf-8"),
                ),
            )

    def load(self, run_id: str) -> Optional[AgentState]:
        """Load state snapshot."""
        rd = self.run_dir(run_id)
        path = rd / "state.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return AgentState.model_validate(data)

        # fallback sqlite
        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        self._ensure_tables(db_path)
        res = self.mcp.sqlite_exec(db_path, "SELECT state_json FROM runs WHERE run_id=?", (run_id,))
        if res.ok and res.data and res.data.get("rows"):
            txt = res.data["rows"][0][0]
            return AgentState.model_validate(json.loads(txt))
        return None

    def _ensure_tables(self, db_path: str) -> None:
        self.mcp.sqlite_exec(
            db_path,
            """
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY,
              status TEXT,
              pending_action TEXT,
              progress_percent INTEGER,
              current_layer TEXT,
              state_json TEXT
            );
            """,
        )
        self.mcp.sqlite_exec(
            db_path,
            """
            CREATE TABLE IF NOT EXISTS job_tracker(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT,
              applied_date TEXT,
              company TEXT,
              job_url TEXT,
              priority TEXT,
              interview_status TEXT
            );
            """,
        )
        self.mcp.sqlite_exec(
            db_path,
            """
            CREATE TABLE IF NOT EXISTS learning_memory(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_key TEXT,
              signal TEXT,
              payload_json TEXT,
              created_at TEXT
            );
            """,
        )
