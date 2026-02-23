from __future__ import annotations

from typing import Any, Dict, List

from careeragent.core.state import AgentState


class RankerAgentService:
    """Description: Deterministic ranker.
    Layer: L5
    Input: jobs_scored
    Output: ranking
    """

    def rank(self, state: AgentState) -> List[Dict[str, Any]]:
        jobs = list(state.jobs_scored)
        jobs.sort(key=lambda x: float(x.get("overall_match_percent") or 0.0), reverse=True)

        ranking: List[Dict[str, Any]] = []
        for i, j in enumerate(jobs[:60], start=1):
            jj = dict(j)
            jj["rank"] = i
            ranking.append(jj)
        return ranking
