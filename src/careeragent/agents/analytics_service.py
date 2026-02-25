from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from careeragent.core.state import AgentState
from careeragent.agents.analytics_schema import AnalyticsReport


class AnalyticsService:
    """
    Description: L9 analytics engine that aggregates InterviewChanceScore vs Actual Outcome.
    Layer: L9
    Input: OrchestrationState (artifacts/meta)
    Output: AnalyticsReport for feedback + ML calibration
    """

    SCORE_BINS: List[Tuple[float, float]] = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01)]

    def build_report(self, *, orchestration_state: AgentState) -> AnalyticsReport:
        """
        Description: Aggregate submission outcomes against predicted interview chance scores.
        Layer: L9
        Input: OrchestrationState
        Output: AnalyticsReport
        """
        submissions: Dict[str, Dict[str, Any]] = orchestration_state.meta.get("submissions", {}) or {}
        status_updates: List[Dict[str, Any]] = orchestration_state.meta.get("status_updates", []) or []

        # 1) Build submission -> latest outcome
        latest_status_by_submission: Dict[str, str] = {}
        for e in status_updates:
            sid = e.get("submission_id")
            if not sid:
                continue
            latest_status_by_submission[sid] = str(e.get("status", "applied"))

        # 2) Resolve predicted scores
        # Preferred: state.meta['job_scores'][job_id] written by matcher step.
        job_scores: Dict[str, Any] = orchestration_state.meta.get("job_scores", {}) or {}

        rows: List[Dict[str, Any]] = []
        outcomes_summary: Dict[str, int] = {}

        for sid, sub in submissions.items():
            job_id = str(sub.get("job_id", ""))
            outcome = latest_status_by_submission.get(sid, "applied")

            score = self._resolve_score(orchestration_state, job_id, job_scores)
            row = {
                "submission_id": sid,
                "job_id": job_id,
                "predicted_interview_chance_score": score,
                "actual_outcome": outcome,
                "is_interview": 1 if outcome == "interviewing" else 0,
            }
            rows.append(row)
            outcomes_summary[outcome] = outcomes_summary.get(outcome, 0) + 1

        # 3) Mean score by outcome
        mean_score_by_outcome: Dict[str, float] = {}
        grouped: Dict[str, List[float]] = {}
        for r in rows:
            grouped.setdefault(r["actual_outcome"], []).append(float(r["predicted_interview_chance_score"] or 0.0))
        for k, vals in grouped.items():
            mean_score_by_outcome[k] = round(sum(vals) / max(1, len(vals)), 4)

        # 4) Interview rate by score bin (calibration-ish)
        interview_rate_by_bin: Dict[str, float] = {}
        for lo, hi in self.SCORE_BINS:
            bucket = [r for r in rows if lo <= float(r["predicted_interview_chance_score"] or 0.0) < hi]
            if not bucket:
                continue
            rate = sum(int(r["is_interview"]) for r in bucket) / len(bucket)
            interview_rate_by_bin[f"{lo:.1f}-{min(hi,1.0):.1f}"] = round(rate, 4)

        return AnalyticsReport(
            total_submissions=len(submissions),
            outcomes_summary=outcomes_summary,
            mean_score_by_outcome=mean_score_by_outcome,
            interview_rate_by_score_bin=interview_rate_by_bin,
            dataset_rows=rows,
        )

    @staticmethod
    def _resolve_score(
        orchestration_state: AgentState,
        job_id: str,
        job_scores: Dict[str, Any],
    ) -> float:
        """
        Description: Resolve interview chance score deterministically from state/meta or artifacts.
        Layer: L9
        Input: state + job_id + job_scores
        Output: float in [0,1]
        """
        # 1) Meta cache
        if job_id in job_scores:
            try:
                return float(job_scores[job_id])
            except Exception:
                pass

        # 2) Try reading MatchReport artifact if present
        art_key = f"match_report_{job_id}"
        ref = orchestration_state.artifacts.get(art_key)
        if ref and ref.path and Path(ref.path).exists():
            try:
                data = json.loads(Path(ref.path).read_text(encoding="utf-8"))
                return float(data.get("interview_chance_score", 0.0))
            except Exception:
                return 0.0

        return 0.0
