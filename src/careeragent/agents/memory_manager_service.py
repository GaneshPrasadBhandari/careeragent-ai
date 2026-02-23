from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from careeragent.core.mcp_client import MCPClient, sqlite_path_from_database_url
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class MemoryManager:
    """Description: Prevent duplicate applications and support lightweight RAG memory.
    Layer: L8

    Storage:
      - Uses Qdrant if configured (payload-based dedupe)
      - Always writes URL hashes into SQLite for deterministic dedupe
    """

    def __init__(self, settings: Settings, mcp: MCPClient) -> None:
        self.s = settings
        self.mcp = mcp

    def already_applied(self, url: str) -> bool:
        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        h = _url_hash(url)
        res = self.mcp.sqlite_exec(db_path, "SELECT 1 FROM learning_memory WHERE signal='applied_url' AND payload_json LIKE ? LIMIT 1", (f"%{h}%",))
        return bool(res.ok and res.data and res.data.get("rows"))

    def mark_applied(self, url: str, *, user_key: str = "default") -> None:
        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        h = _url_hash(url)
        payload = {"url": url, "hash": h}
        self.mcp.sqlite_exec(
            db_path,
            "INSERT INTO learning_memory(user_key, signal, payload_json, created_at) VALUES(?,?,?,datetime('now'))",
            (user_key, "applied_url", str(payload)),
        )

    def filter_duplicates(self, state: AgentState, urls: List[str]) -> List[str]:
        out: List[str] = []
        for u in urls:
            if self.already_applied(u):
                state.log_eval(f"[L8] duplicate prevented: {u}")
                continue
            out.append(u)
        return out
