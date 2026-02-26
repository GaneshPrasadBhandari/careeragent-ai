
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from careeragent.core.state import AgentState


@dataclass
class QuotaManager:
    """
    Description: Tracks quota errors for external APIs and blocks runs.
    Layer: L0
    """
    serper_quota_exceeded: bool = False
    last_error: Optional[str] = None

    def handle_serper_response(self, *, state: AgentState, step_id: str, status_code: int, tool_name: str, error_detail: str) -> bool:
        if status_code == 403:
            self.serper_quota_exceeded = True
            self.last_error = error_detail
            state.status = "blocked"
            state.meta["run_failure_code"] = "API_FAILURE"
            state.meta["run_failure_provider"] = "serper"
            state.meta["run_failure_detail"] = error_detail
            # mark step blocked if possible
            try:
                state.end_step(step_id, status="blocked", output_ref={"error": "quota_exceeded"}, message="serper_quota_exceeded")
            except Exception:
                pass
            return True
        return False


class HealthService:
    """
    Description: Local health + tracing wiring.
    Layer: L0
    """
    def __init__(self) -> None:
        self.quota = QuotaManager()

    def load_env(self, *, dotenv_path: str) -> None:
        # No-op: settings already load .env via pydantic-settings in your repo.
        return

    def enable_langsmith_tracing(self, *, project: str) -> None:
        # No-op: optional integration later.
        return
