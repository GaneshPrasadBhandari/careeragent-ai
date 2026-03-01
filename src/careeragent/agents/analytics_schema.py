from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field




class FunnelAuditReport(BaseModel):
    """Run-level discovered→ranked→approved→attempted→submitted funnel with blocker taxonomy."""

    discovered: int = 0
    ranked: int = 0
    approved: int = 0
    attempted: int = 0
    submitted: int = 0
    conversion_rates: Dict[str, float] = Field(default_factory=dict)
    blocker_taxonomy: Dict[str, int] = Field(default_factory=dict)


class SourceTelemetryReport(BaseModel):
    """Per-source health telemetry and quota utilization for discovery pipelines."""

    source_counts: Dict[str, int] = Field(default_factory=dict)
    source_errors: Dict[str, int] = Field(default_factory=dict)
    source_quota_targets: Dict[str, int] = Field(default_factory=dict)


class AnalyticsReport(BaseModel):
    """
    Description: L9 analytics artifact aggregating InterviewChanceScore vs Actual Outcome.
    Layer: L9
    Input: OrchestrationState (submissions + match scores + status updates)
    Output: Summary metrics + training dataset rows for future ML calibration
    """

    total_submissions: int
    outcomes_summary: Dict[str, int] = Field(default_factory=dict)

    mean_score_by_outcome: Dict[str, float] = Field(default_factory=dict)
    interview_rate_by_score_bin: Dict[str, float] = Field(default_factory=dict)

    dataset_rows: List[Dict[str, Any]] = Field(default_factory=list)
    funnel_audit: FunnelAuditReport = Field(default_factory=FunnelAuditReport)
    source_telemetry: SourceTelemetryReport = Field(default_factory=SourceTelemetryReport)
