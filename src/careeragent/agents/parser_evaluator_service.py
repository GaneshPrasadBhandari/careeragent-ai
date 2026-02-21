
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from careeragent.agents.parser_agent_service import ExtractedResume


class EvaluationEvent(BaseModel):
    """
    Description: Evaluation output for recursive gate decisions.
    Layer: L3
    Input: generator output
    Output: score + feedback + gate decision support
    """
    model_config = ConfigDict(extra="ignore")
    evaluation_score: float = 0.0
    threshold: float = 0.0
    feedback: List[str] = Field(default_factory=list)


def _ats_structure_score(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    if "summary" in t: score += 0.15
    if "skills" in t: score += 0.20
    if "experience" in t: score += 0.25
    if "education" in t: score += 0.10
    if "-" in text or "•" in text: score += 0.15
    if len(text) > 1200: score += 0.15
    return max(0.0, min(1.0, score))


class ParserEvaluatorService:
    """
    Description: Critique Parser output for completeness + ATS readiness (soft gate).
    Layer: L3
    Input: raw_text + ExtractedResume
    Output: EvaluationEvent (score + feedback)
    """
    def evaluate(
        self,
        *,
        orchestration_state: Any,
        raw_text: str,
        extracted: ExtractedResume,
        target_id: str,
        threshold: float = 0.55,
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> Any:
        feedback: List[str] = []

        contact = extracted.contact or None
        has_contact = bool(getattr(contact, "email", None) or getattr(contact, "phone", None))
        has_skills = len(extracted.skills or []) >= 6
        has_exp = bool(extracted.experience and len(extracted.experience) > 0)
        ats = _ats_structure_score(raw_text)

        # weights (so missing contact doesn't kill automation)
        w_contact = 0.15
        w_skills = 0.35
        w_exp = 0.35
        w_ats = 0.15

        s_contact = 1.0 if has_contact else 0.0
        s_skills = min(1.0, len(extracted.skills or []) / 12.0)
        s_exp = 1.0 if has_exp else 0.0
        s_ats = ats

        score = (w_contact*s_contact) + (w_skills*s_skills) + (w_exp*s_exp) + (w_ats*s_ats)
        score = max(0.0, min(1.0, float(score)))

        if not has_contact:
            feedback.append("Contact info missing (email/phone). Optional: paste improved resume to continue with stronger ATS fit.")
        if not has_exp:
            feedback.append("Experience section unclear. Add an Experience heading with bullet points.")
        if len(extracted.skills or []) < 6:
            feedback.append("Skills extraction low. Add a clearer Skills section with comma-separated skills.")
        if ats < 0.5:
            feedback.append("ATS structure weak. Add headings: Summary, Skills, Experience, Education; use bullets.")

        # Record in orchestration state (your OrchestrationState already has record_evaluation)
        ev = orchestration_state.record_evaluation(
            layer_id="L3",
            target_id=target_id,
            generator_agent="parser_agent_service",
            evaluator_agent="parser_evaluator_service",
            evaluation_score=score,
            threshold=threshold,
            feedback=feedback,
            retry_count=retry_count,
            max_retries=max_retries,
            interview_chance={
                "weights": {"w1_skill_overlap": 0.45, "w2_experience_alignment": 0.35, "w3_ats_score": 0.20},
                "components": {"skill_overlap": 0.0, "experience_alignment": 0.0, "ats_score": float(ats), "market_competition_factor": 1.0},
                "interview_chance_score": float((0.20*ats)),  # parser-stage proxy only
            },
        )
        return ev
