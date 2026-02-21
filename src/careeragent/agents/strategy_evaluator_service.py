from __future__ import annotations

from typing import List

from careeragent.orchestration.state import OrchestrationState
from careeragent.agents.strategy_agent_schema import PivotStrategy
from careeragent.agents.matcher_agent_schema import MatchReport


class StrategyEvaluatorService:
    """
    Description: L5 evaluator twin for PivotStrategy (Recursive Gate).
    Layer: L5
    Input: MatchReport + PivotStrategy
    Output: EvaluationEvent logged to OrchestrationState
    """

    def evaluate(
        self,
        *,
        orchestration_state: OrchestrationState,
        match_report: MatchReport,
        strategy: PivotStrategy,
        target_id: str,
        threshold: float = 0.80,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Description: Validate that strategy is actionable given the match severity.
        Layer: L5
        Input: state + match_report + strategy
        Output: EvaluationEvent
        """
        feedback: List[str] = []
        score = 1.0

        m = float(match_report.overall_match_percent)
        n_items = len(strategy.action_items or [])

        if m < 70 and strategy.posture != "pivot":
            score -= 0.35
            feedback.append("Posture mismatch: for match < 70%, posture should be 'pivot'.")

        if m < 70:
            if n_items < 3:
                score -= 0.40
                feedback.append("Strategy too thin for low match: add more ActionItems (target 3–5) with concrete steps.")
        else:
            if n_items < 2:
                score -= 0.25
                feedback.append("Strategy too thin: add at least 2 ActionItems for optimization.")

        # Each action item should include how_to_execute steps.
        if any((not it.how_to_execute) for it in (strategy.action_items or [])):
            score -= 0.20
            feedback.append("ActionItems must include step-by-step 'how_to_execute' bullets.")

        score = max(0.0, min(1.0, score))

        return orchestration_state.record_evaluation(
            layer_id="L5",
            target_id=target_id,
            generator_agent="strategy_agent_service",
            evaluator_agent="strategy_evaluator_service",
            evaluation_score=float(score),
            threshold=float(threshold),
            feedback=feedback,
            retry_count=int(retry_count),
            max_retries=int(max_retries),
            interview_chance=None,
        )
