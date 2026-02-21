from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsReport(BaseModel):
    """
    Description: L9 analytics artifact aggregating InterviewChanceScore vs Actual Outcome.
    Layer: L9
    Input: OrchestrationState (submissions + match scores + status updates)
    Output: Summary metrics + training dataset rows for future ML calibration
    """

    model_config = ConfigDict(extra="forbid")

    total_submissions: int
    outcomes_summary: Dict[str, int] = Field(default_factory=dict)

    mean_score_by_outcome: Dict[str, float] = Field(default_factory=dict)
    interview_rate_by_score_bin: Dict[str, float] = Field(default_factory=dict)

    dataset_rows: List[Dict[str, Any]] = Field(default_factory=list)
