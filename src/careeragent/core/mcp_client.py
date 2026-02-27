from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

    def _remote_invoke(self, tool: str, args: Dict[str, Any]) -> Optional[MCPResult]:
        """Best-effort remote MCP invocation.

        Description:
          If MCP_SERVER_URL is configured, attempt a simple HTTP invocation.
          We intentionally keep this flexible: different MCP servers expose
          different shapes. We try a very small contract first:
            POST {MCP_SERVER_URL}/invoke  {tool: str, args: dict}

        Layer: L8
        Input: tool + args
        Output: MCPResult or None if not available
        """

        base = (self.s.MCP_SERVER_URL or "").rstrip("/")
        if not base:
            return None

        # Safety guard: old CareerOS backend endpoint is not valid for this repo.
        if "careeros-backend" in base:
            return MCPResult(ok=False, error="MCP_SERVER_URL points to legacy CareerOS backend; using local fallback")

        try:
            import httpx

            candidate_urls = [f"{base}/invoke"]
            if not base.endswith("/mcp"):
                candidate_urls.append(f"{base}/mcp/invoke")

            with httpx.Client(timeout=self.s.MAX_HTTP_SECONDS) as client:
                last_error = None
                for url in candidate_urls:
                    r = client.post(url, json={"tool": tool, "args": args})
                    if r.status_code == 404:
                        last_error = f"{r.status_code} {r.text[:200]}"
                        continue
                    if r.status_code >= 400:
                        return MCPResult(ok=False, error=f"MCP remote {tool} failed: {r.status_code} {r.text[:200]}")
                    return MCPResult(ok=True, data=r.json())
            if last_error:
                return MCPResult(ok=False, error=f"MCP remote {tool} failed: {last_error}")
            return MCPResult(ok=False, error=f"MCP remote {tool} failed")
        except Exception as e:
            # Fall back to local implementation.
            return MCPResult(ok=False, error=f"MCP remote unavailable: {e}")

    # ------------------------
    # Files
    # ------------------------
    def write_file(self, path: str, content: bytes) -> MCPResult:
        """Write bytes to disk."""
        try:
            import base64

            _ = self._remote_invoke("write_file", {"path": path, "content_b64": base64.b64encode(content).decode("ascii")})
        except Exception:
            pass
        # If remote is configured but fails, continue locally.
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
            return MCPResult(ok=True, data={"path": str(p)})
        except Exception as e:
            return MCPResult(ok=False, error=str(e))

    def read_file(self, path: str) -> MCPResult:
        """Read bytes from disk."""
        remote = self._remote_invoke("read_file", {"path": path})
        if remote and remote.ok and isinstance(remote.data, dict) and remote.data.get("content_b64"):
            try:
                import base64

                return MCPResult(ok=True, data=base64.b64decode(remote.data["content_b64"]))
            except Exception:
                pass
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
        # Best-effort remote DB execution if MCP server supports it.
        # We do not require this for local development.
        remote = self._remote_invoke("sqlite_exec", {"db_path": db_path, "sql": sql, "params": list(params)})
        if remote and remote.ok and isinstance(remote.data, dict) and ("rows" in remote.data or "rowcount" in remote.data):
            return MCPResult(ok=True, data=remote.data)
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
