
from __future__ import annotations

import re
from typing import Any, List
from careeragent.agents.parser_agent_service import ExtractedResume


def _ats_structure_score(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    for h in ["summary","skills","experience","education","projects"]:
        if h in t:
            score += 0.14
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", t):
        score += 0.15
    if "-" in text or "•" in text:
        score += 0.15
    if len(text) > 1200:
        score += 0.15
    return max(0.0, min(1.0, score))


class ParserEvaluatorService:
    """
    Description: Evaluates extracted profile quality (SOFT gate).
    Layer: L3
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
        max_retries: int = 0,
    ) -> Any:
        fb: List[str] = []
        contact_ok = bool(extracted.contact.email or extracted.contact.phone)
        skills_ok = len(extracted.skills or []) >= 6
        exp_ok = bool(extracted.experience and extracted.experience[0].bullets)
        ats = _ats_structure_score(raw_text)

        score = (0.15*(1.0 if contact_ok else 0.0) +
                 0.40*min(1.0, len(extracted.skills or [])/12.0) +
                 0.30*(1.0 if exp_ok else 0.0) +
                 0.15*ats)

        if not contact_ok:
            fb.append("Contact missing. Continue anyway; but adding email/phone improves ATS + recruiter trust.")
        if not exp_ok:
            fb.append("No bullet points detected. Continue anyway; but bullets improve ATS scanability.")
        if not skills_ok:
            fb.append("Skills list looks thin. Continue anyway; add more relevant tools/keywords.")
        if ats < 0.5:
            fb.append("ATS structure weak. Add headings and bullets.")

        return orchestration_state.record_evaluation(
            layer_id="L3",
            target_id=target_id,
            generator_agent="parser_agent_service",
            evaluator_agent="parser_evaluator_service",
            evaluation_score=float(score),
            threshold=float(threshold),
            feedback=fb,
            retry_count=int(retry_count),
            max_retries=int(max_retries),
            interview_chance={
                "weights": {"w1_skill_overlap": 0.45, "w2_experience_alignment": 0.35, "w3_ats_score": 0.20},
                "components": {"skill_overlap": 0.0, "experience_alignment": 0.0, "ats_score": float(ats), "market_competition_factor": 1.0},
                "interview_chance_score": float(0.20*ats),
            },
        )
