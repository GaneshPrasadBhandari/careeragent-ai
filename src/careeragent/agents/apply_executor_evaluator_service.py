from __future__ import annotations

from typing import List

from careeragent.orchestration.state import AgentState
from careeragent.agents.apply_executor_schema import ApplicationSubmission


class ApplyExecutorEvaluatorService:
    """
    Description: L7 evaluator twin that verifies submission integrity and state recording.
    Layer: L7
    Input: AgentState + ApplicationSubmission
    Output: EvaluationEvent logged to AgentState (Recursive Gate compatible)
    """

    def evaluate(
        self,
        *,
        orchestration_state: AgentState,
        submission: ApplicationSubmission,
        target_id: str,
        threshold: float = 0.90,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Description: Validate that submission_id and timestamp are present and stored in state.
        Layer: L7
        Input: state + submission
        Output: EvaluationEvent
        """
        feedback: List[str] = []
        score = 1.0

        if not submission.submission_id:
            score -= 0.60
            feedback.append("Missing submission_id: executor must generate a stable submission identifier.")

        if not submission.submitted_at_utc:
            score -= 0.30
            feedback.append("Missing timestamp: executor must record submitted_at_utc.")

        subs = orchestration_state.meta.get("submissions", {})
        if submission.submission_id not in subs:
            score -= 0.40
            feedback.append("State recording missing: submission not found under state.meta['submissions'].")

        # Completed status is expected only after success.
        if orchestration_state.status != "completed":
            score -= 0.25
            feedback.append("RunStatus not updated: state.status should be 'completed' after L7 success.")

        score = max(0.0, min(1.0, float(score)))

        return orchestration_state.record_evaluation(
            layer_id="L7",
            target_id=target_id,
            generator_agent="apply_executor_service",
            evaluator_agent="apply_executor_evaluator_service",
            evaluation_score=score,
            threshold=float(threshold),
            feedback=feedback,
            retry_count=int(retry_count),
            max_retries=int(max_retries),
            interview_chance=None,
        )
