from __future__ import annotations

import math
from typing import Any, Dict

from careeragent.core.state import AgentState


class GovernanceAuditor:
    """Description: Monitor token costs and enforce compliance flags.
    Layer: L9
    """

    def estimate_tokens(self, text: str) -> int:
        # rough: 4 chars per token
        return int(math.ceil(len(text) / 4.0))

    def add_tokens(self, state: AgentState, prompt: str, completion: str = "") -> None:
        state.token_used_est += self.estimate_tokens(prompt) + self.estimate_tokens(completion)
        if state.token_used_est > state.token_budget:
            state.status = "failed"
            state.pending_action = None
            state.log_eval(f"[L9] token budget exceeded: {state.token_used_est} > {state.token_budget}")

    def finalize(self, state: AgentState) -> None:
        # record summary
        state.meta.setdefault("governance", {})
        state.meta["governance"].update(
            {
                "token_used_est": state.token_used_est,
                "token_budget": state.token_budget,
                "robots_violations": state.robots_violations,
            }
        )
