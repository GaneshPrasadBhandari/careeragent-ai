from __future__ import annotations

import json
import math
from typing import Optional

from careeragent.core.mcp_client import MCPClient, sqlite_path_from_database_url
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState


class GovernanceAuditor:
    """Monitor token costs and persist final governance summary."""

    def __init__(self, settings: Optional[Settings] = None, mcp: Optional[MCPClient] = None) -> None:
        self.s = settings
        self.mcp = mcp

    def estimate_tokens(self, text: str) -> int:
        return int(math.ceil(len(text) / 4.0))

    def add_tokens(self, state: AgentState, prompt: str, completion: str = "") -> None:
        state.token_used_est += self.estimate_tokens(prompt) + self.estimate_tokens(completion)
        if state.token_used_est > state.token_budget:
            state.status = "failed"
            state.pending_action = None
            state.log_eval(f"[L9] token budget exceeded: {state.token_used_est} > {state.token_budget}")

    def finalize(self, state: AgentState) -> None:
        weights = state.meta.get("interview_chance_weights") or {"w1_skill_overlap": 0.45, "w2_experience_alignment": 0.35, "w3_ats_score": 0.20}
        signals = {
            "retry_count": state.retry_count,
            "active_persona": state.active_persona_id,
            "recent_eval_logs": state.evaluation_logs[-8:],
        }
        state.meta.setdefault("governance", {})
        state.meta["governance"].update(
            {
                "token_used_est": state.token_used_est,
                "token_budget": state.token_budget,
                "robots_violations": state.robots_violations,
                "final_interview_chance_weights": weights,
                "self_learning_signals": signals,
            }
        )

        if self.s and self.mcp:
            db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
            payload = {"run_id": state.run_id, "weights": weights, "signals": signals}
            self.mcp.sqlite_exec(
                db_path,
                "INSERT INTO learning_memory(user_key, signal, payload_json, created_at) VALUES(?,?,?,datetime('now'))",
                ("default", "governance_summary", json.dumps(payload)),
            )
