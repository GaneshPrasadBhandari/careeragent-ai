from __future__ import annotations

from typing import List, Tuple

from careeragent.orchestration.state import OrchestrationState, InterviewChanceBreakdown
from careeragent.agents.matcher_agent_schema import JobDescription, MatchReport
from careeragent.agents.parser_agent_service import ExtractedResume
from careeragent.agents.matcher_agent_service import MatcherAgentService


class MatchEvaluatorService:
    """
    Description: L4 evaluator twin that verifies scoring math consistency and report integrity.
    Layer: L4
    Input: Resume + Job + MatchReport
    Output: EvaluationEvent logged to OrchestrationState
    """

    def evaluate(
        self,
        *,
        orchestration_state: OrchestrationState,
        resume: ExtractedResume,
        job: JobDescription,
        report: MatchReport,
        target_id: str,
        threshold: float = 0.80,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Description: Validate match report consistency and math.
        Layer: L4
        Input: state + inputs + report
        Output: EvaluationEvent
        """
        feedback: List[str] = []
        score = 1.0

        # Recompute expected report deterministically and compare key fields.
        matcher = MatcherAgentService()
        expected = matcher.match(resume=resume, job=job, orchestration_state=orchestration_state)

        # InterviewChanceScore should match very closely.
        diff = abs(float(expected.interview_chance_score) - float(report.interview_chance_score))
        if diff > 1e-6:
            score -= 0.45
            feedback.append(
                f"Scoring math mismatch: expected interview_chance_score={expected.interview_chance_score:.6f}, got {report.interview_chance_score:.6f}."
            )

        # Components should align within tolerance.
        comps = ["skill_overlap", "experience_alignment", "ats_score", "market_competition_factor"]
        for c in comps:
            d = abs(float(getattr(expected.components, c)) - float(getattr(report.components, c)))
            if d > 1e-6:
                score -= 0.10
                feedback.append(f"Component mismatch for {c}: expected {getattr(expected.components,c):.6f}, got {getattr(report.components,c):.6f}.")

        # Missing required skills must be subset of required skills.
        req = set([s.strip().lower() for s in job.required_skills])
        miss = set([s.strip().lower() for s in report.missing_required_skills])
        if not miss.issubset(req):
            score -= 0.20
            feedback.append("Integrity issue: missing_required_skills contains skills not present in job.required_skills.")

        score = max(0.0, min(1.0, score))

        # Use same InterviewChanceBreakdown stored in evaluation for observability.
        interview = InterviewChanceBreakdown(
            weights=orchestration_state.meta.get("weights_override") or expected  # not used, placeholder to keep types happy
        ) if False else None  # keep deterministic; we’ll attach breakdown via state formula in next layers

        return orchestration_state.record_evaluation(
            layer_id="L4",
            target_id=target_id,
            generator_agent="matcher_agent_service",
            evaluator_agent="matcher_evaluator_service",
            evaluation_score=float(score),
            threshold=float(threshold),
            feedback=feedback,
            retry_count=int(retry_count),
            max_retries=int(max_retries),
            interview_chance=None,
        )
