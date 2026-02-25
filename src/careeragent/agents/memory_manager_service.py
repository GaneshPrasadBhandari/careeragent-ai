from __future__ import annotations

import hashlib
from typing import List

from careeragent.core.mcp_client import MCPClient, sqlite_path_from_database_url
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class MemoryManager:
    """Prevent duplicate applications via SQLite job_tracker + learning memory."""

    def __init__(self, settings: Settings, mcp: MCPClient) -> None:
        self.s = settings
        self.mcp = mcp

    def already_applied(self, url: str) -> bool:
        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        h = _url_hash(url)
        # Primary dedupe source: job_tracker.
        res = self.mcp.sqlite_exec(db_path, "SELECT 1 FROM job_tracker WHERE job_url_hash=? LIMIT 1", (h,))
        if bool(res.ok and res.data and res.data.get("rows")):
            return True
        # Backward compatibility fallback.
        old = self.mcp.sqlite_exec(db_path, "SELECT 1 FROM learning_memory WHERE signal='applied_url' AND payload_json LIKE ? LIMIT 1", (f"%{h}%",))
        return bool(old.ok and old.data and old.data.get("rows"))

    def mark_applied(self, url: str, *, user_key: str = "default") -> None:
        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        h = _url_hash(url)
        self.mcp.sqlite_exec(
            db_path,
            "INSERT OR IGNORE INTO job_tracker(run_id, applied_date, company, job_url, priority, interview_status, job_url_hash) VALUES(?,?,?,?,?,?,?)",
            ("memory", "", "", url, "low", "applied", h),
        )
        self.mcp.sqlite_exec(
            db_path,
            "INSERT INTO learning_memory(user_key, signal, payload_json, created_at) VALUES(?,?,?,datetime('now'))",
            (user_key, "applied_url", str({"url": url, "hash": h})),
        )

    def filter_duplicates(self, state: AgentState, urls: List[str]) -> List[str]:
        out: List[str] = []
        for u in urls:
            if self.already_applied(u):
                state.log_eval(f"[L8] duplicate prevented: {u}")
                continue
            out.append(u)
        return out
