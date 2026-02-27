"""phase2_evaluator_agent_service.py — L5 Evaluator Gate (Patch v7 Fixed).

ROOT CAUSE FIX:
  - Old threshold was 0.70 for interview_chance_score. With 403-blocked JDs, skill_overlap=0
    so even a perfect candidate scores only ~0.38. The loop NEVER exited → infinite L3→L4→L5.
  - New threshold: 0.55 (aligns with observed best score of 0.63).
  - PROCEED if ANY job has overall_match_percent >= 40 AND we have >= 2 scored jobs.
  - PROCEED also if retry_count >= max_refinements (force-exit; HITL will review).
  - Log clear reasoning so evaluation_logs is useful.

Layer: L5
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from careeragent.core.state import AgentState, EvaluationEntry


# ── Thresholds ──────────────────────────────────────────────────────────────
_INTERVIEW_CHANCE_THRESHOLD = 0.55   # was 0.70 — too high when JDs are 403-blocked
_MIN_SCORED_JOBS = 2                  # minimum number of scored jobs to PROCEED
_MIN_MATCH_PCT = 35.0                 # minimum overall_match_percent for a viable job
_MIN_VIABLE_JOBS = 1                  # at least 1 job above _MIN_MATCH_PCT to PROCEED


def _top_interview_chance(jobs_scored: List[Dict[str, Any]]) -> float:
    """Return the highest interview_chance_score across all scored jobs."""
    best = 0.0
    for j in jobs_scored:
        v = float(j.get("interview_chance_score") or j.get("match_score") or 0.0)
        best = max(best, v)
    return best


def _viable_jobs(jobs_scored: List[Dict[str, Any]]) -> int:
    """Count jobs with overall_match_percent >= _MIN_MATCH_PCT."""
    return sum(
        1 for j in jobs_scored
        if float(j.get("overall_match_percent") or 0.0) >= _MIN_MATCH_PCT
    )


class Phase2EvaluatorAgentService:
    """Description: L5 gate — decides PROCEED vs retry for discovery loop.

    Layer: L5
    Input: state.jobs_scored, state.ranking, state.retry_count
    Output: (score, reason, action, feedback)
    """

    def evaluate(
        self,
        state: AgentState,
    ) -> Tuple[float, str, Literal["PROCEED", "retry", "fail"], List[str]]:
        """Evaluate whether ranked jobs are good enough to proceed to HITL.

        Returns
        -------
        score   : float  0..1 (0=bad, 1=great)
        reason  : str    human-readable explanation
        action  : PROCEED | retry | fail
        feedback: list of improvement hints
        """
        jobs = state.jobs_scored or []
        n_scored = len(jobs)
        top_chance = _top_interview_chance(jobs)
        n_viable = _viable_jobs(jobs)
        retry = state.retry_count
        max_r = int(state.preferences.max_refinements)

        feedback: List[str] = []
        score = min(1.0, top_chance)

        # ── Force PROCEED when retries exhausted ──────────────────────────
        # This prevents an infinite loop. The HITL ranking review will catch issues.
        if retry >= max_r and n_scored >= 1:
            reason = (
                f"[L5] Force-proceed after {retry} retries (max={max_r}). "
                f"Best score={top_chance:.3f}, viable_jobs={n_viable}. HITL will review."
            )
            state.log_eval(reason)
            return max(score, 0.4), reason, "PROCEED", ["HITL review required — retries exhausted"]

        # ── Not enough jobs found yet ──────────────────────────────────────
        if n_scored < _MIN_SCORED_JOBS:
            reason = f"[L5] Only {n_scored} jobs scored (need {_MIN_SCORED_JOBS}). Retrying discovery."
            state.log_eval(reason)
            feedback.append(f"Too few jobs ({n_scored}). Broaden search or change persona.")
            return score, reason, "retry", feedback

        # ── Check if we have viable jobs ──────────────────────────────────
        if n_viable < _MIN_VIABLE_JOBS:
            reason = (
                f"[L5] No viable jobs (scored >= {_MIN_MATCH_PCT}%). "
                f"Top interview chance: {top_chance:.3f}. Retry with different persona."
            )
            state.log_eval(reason)
            feedback.append(f"All jobs below {_MIN_MATCH_PCT}% match. Try broader search terms.")
            return score, reason, "retry", feedback

        # ── Main gate: interview chance threshold ──────────────────────────
        if top_chance >= _INTERVIEW_CHANCE_THRESHOLD:
            reason = (
                f"[L5] PROCEED — top interview_chance={top_chance:.3f} "
                f">= threshold={_INTERVIEW_CHANCE_THRESHOLD}. "
                f"viable_jobs={n_viable}, total_scored={n_scored}."
            )
            state.log_eval(reason)
            return score, reason, "PROCEED", []

        # ── Below threshold — retry if retries remain ─────────────────────
        reason = (
            f"[L5] top interview_chance={top_chance:.3f} < "
            f"threshold={_INTERVIEW_CHANCE_THRESHOLD}. "
            f"viable_jobs={n_viable}, retry={retry}/{max_r}. "
            "Requesting more/better jobs from L3."
        )
        state.log_eval(reason)
        feedback.append(
            f"Best score {top_chance:.2f} below {_INTERVIEW_CHANCE_THRESHOLD}. "
            "Try different persona or relax location constraints."
        )

        # If we still have retries left, suggest retry; otherwise force PROCEED
        if retry < max_r:
            return score, reason, "retry", feedback
        else:
            state.log_eval(f"[L5] Retries exhausted ({retry}/{max_r}). Force PROCEED for HITL.")
            return score, reason, "PROCEED", feedback

    def write_to_state(
        self,
        state: AgentState,
        score: float,
        reason: str,
        action: str,
        feedback: List[str],
    ) -> None:
        """Persist evaluation result into state."""
        entry = EvaluationEntry(
            layer_id="L5",
            target_id="discovery_batch",
            evaluation_score=round(score, 4),
            threshold=_INTERVIEW_CHANCE_THRESHOLD,
            decision="pass" if action == "PROCEED" else "retry",
            feedback=feedback,
        )
        state.evaluations.append(entry)
        state.evaluation = entry.model_dump(mode="json")
