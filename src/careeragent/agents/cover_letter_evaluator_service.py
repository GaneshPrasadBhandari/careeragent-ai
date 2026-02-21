from __future__ import annotations

import re
from typing import List

from careeragent.orchestration.state import OrchestrationState
from careeragent.agents.cover_letter_agent_schema import CoverLetterDraft
from careeragent.agents.parser_agent_service import ExtractedResume
from careeragent.agents.matcher_agent_schema import JobDescription, MatchReport


class CoverLetterEvaluatorService:
    """
    Description: L6 evaluator twin for cover letter quality + compliance.
    Layer: L6
    Input: Resume + Job + MatchReport + Draft
    Output: EvaluationEvent logged to OrchestrationState (Recursive Gate)
    """

    def evaluate(
        self,
        *,
        orchestration_state: OrchestrationState,
        resume: ExtractedResume,
        job: JobDescription,
        match_report: MatchReport,
        draft: CoverLetterDraft,
        target_id: str,
        threshold: float = 0.80,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Description: Evaluate cover letter for professional tone and required fields.
        Layer: L6
        Input: state + inputs + draft
        Output: EvaluationEvent
        """
        feedback: List[str] = []
        score = 1.0

        txt = draft.body or ""
        words = len(re.findall(r"\w+", txt))

        # Must reference role + company
        if job.role_title.lower() not in txt.lower():
            score -= 0.25
            feedback.append("Missing role title: explicitly mention the role you’re applying for.")
        if job.company.lower() not in txt.lower():
            score -= 0.25
            feedback.append("Missing company name: explicitly mention the company.")

        # Professional contact block (at least email if known)
        if resume.contact.email and resume.contact.email not in txt:
            score -= 0.30
            feedback.append("Contact block missing: include your email (and phone/link if available) in the header.")

        # Should include at least 2 highlighted skills (if available)
        if draft.highlighted_skills:
            hits = sum(1 for s in draft.highlighted_skills[:5] if s.lower() in txt.lower())
            if hits < 2:
                score -= 0.20
                feedback.append("Skill evidence low: weave 2–3 matched skills into the opening paragraph.")

        # Length control (ATS-friendly)
        if words > 450:
            score -= 0.15
            feedback.append("Too long: keep cover letter under ~450 words (tight 3–4 paragraphs).")

        # Tone check (simple heuristic)
        forbidden = ["desperate", "please give me", "any job", "kindly do the needful"]
        if any(f in txt.lower() for f in forbidden):
            score -= 0.20
            feedback.append("Tone issue: remove informal/pleading phrasing; keep it confident and factual.")

        score = max(0.0, min(1.0, score))

        return orchestration_state.record_evaluation(
            layer_id="L6",
            target_id=target_id,
            generator_agent="cover_letter_service",
            evaluator_agent="cover_letter_evaluator_service",
            evaluation_score=float(score),
            threshold=float(threshold),
            feedback=feedback,
            retry_count=int(retry_count),
            max_retries=int(max_retries),
            interview_chance=None,
        )
