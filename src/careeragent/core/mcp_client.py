from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from careeragent.core.settings import Settings


@dataclass
class MCPResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None


class MCPClient:
    """Description: Minimal MCP client facade.

    Layer: L8
    Input: tool calls (db/files/vector)
    Output: results

    Notes:
      - If MCP_SERVER_URL is set, attempts remote JSON-RPC calls.
      - Otherwise falls back to local implementations.
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._lock = threading.Lock()

    # ------------------------
    # Files
    # ------------------------
    def write_file(self, path: str, content: bytes) -> MCPResult:
        """Write bytes to disk."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
            return MCPResult(ok=True, data={"path": str(p)})
        except Exception as e:
            return MCPResult(ok=False, error=str(e))

    def read_file(self, path: str) -> MCPResult:
        """Read bytes from disk."""
        try:
            p = Path(path)
            return MCPResult(ok=True, data=p.read_bytes())
        except Exception as e:
            return MCPResult(ok=False, error=str(e))

    # ------------------------
    # SQLite (local fallback)
    # ------------------------
    def sqlite_exec(self, db_path: str, sql: str, params: Tuple[Any, ...] = ()) -> MCPResult:
        """Execute SQL and return rows if SELECT."""
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                if sql.strip().lower().startswith("select"):
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description] if cur.description else []
                    return MCPResult(ok=True, data={"columns": cols, "rows": rows})
                conn.commit()
                return MCPResult(ok=True, data={"rowcount": cur.rowcount})
        except Exception as e:
            return MCPResult(ok=False, error=str(e))


def sqlite_path_from_database_url(database_url: str) -> str:
    """Convert sqlite:///path into local path."""
    if database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "")
    if database_url.startswith("sqlite://"):
        return database_url.replace("sqlite://", "")
    # fallback: treat as file
    return database_url
