from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, InterviewChanceBreakdown, InterviewChanceComponents, InterviewChanceWeights
from careeragent.nlp.skills import compute_jd_alignment, normalize_skill
from careeragent.tools.llm_tools import GeminiClient


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9+.#]", (text or "").lower()) if t and len(t) > 1]


class MatcherAgentService:
    """Description: L4 matcher with semantic interview-chance scoring.

    Layer: L4
    Input: extracted_profile + jobs_raw full text
    Output: jobs_scored list
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.s = settings or Settings()
        self.gemini = GeminiClient(self.s)

    def _semantic_fit(self, *, profile: Dict[str, Any], job: Dict[str, Any], fallback: float) -> float:
        if not self.s.GEMINI_API_KEY:
            return fallback
        jd = (job.get("full_text_md") or job.get("snippet") or "")[:7000]
        prompt = (
            "You are a hiring manager scoring semantic candidate-job fit. "
            "Return STRICT JSON {score: float 0..1, reason: string}. "
            "Score based on real capability alignment, not raw keyword counting.\n\n"
            f"PROFILE_JSON: {profile}\n\nJOB_TITLE: {job.get('title')}\nJOB_TEXT:\n{jd}"
        )
        j = self.gemini.generate_json(prompt, temperature=0.1, max_tokens=350)
        if not isinstance(j, dict):
            return fallback
        try:
            return max(0.0, min(1.0, float(j.get("score"))))
        except Exception:
            return fallback

    def score_jobs(self, state: AgentState) -> List[Dict[str, Any]]:
        prof = state.extracted_profile or {}
        resume_skills_raw = prof.get("skills") or []
        resume_skills = [normalize_skill(s) for s in resume_skills_raw if s]
        weights = InterviewChanceWeights()

        out: List[Dict[str, Any]] = []
        for j in state.jobs_raw:
            jd_text = (j.get("full_text_md") or j.get("snippet") or "")
            jd_title = str(j.get("title") or "")

            scorecard = compute_jd_alignment(jd_text=jd_text, resume_skills=resume_skills)
            skill_overlap = scorecard.jd_alignment_percent / 100.0

            toks = set(_tokenize(jd_text + " " + jd_title))
            exp_align = 0.25
            if any(k in toks for k in ["architect", "solution", "stakeholder", "roadmap", "strategy"]):
                exp_align = 0.65
            if any(k in toks for k in ["genai", "llm", "rag", "langchain", "langgraph", "azure", "openai"]):
                exp_align = min(1.0, exp_align + 0.2)
            if any(k in toks for k in ["mlops", "cicd", "docker", "kubernetes", "terraform", "mlflow"]):
                exp_align = min(1.0, exp_align + 0.15)

            ats_score = min(1.0, 0.05 + skill_overlap)
            semantic_exp_align = self._semantic_fit(profile=prof, job=j, fallback=exp_align)

            breakdown = InterviewChanceBreakdown(
                weights=weights,
                components=InterviewChanceComponents(
                    skill_overlap=round(skill_overlap, 4),
                    experience_alignment=round(semantic_exp_align, 4),
                    ats_score=round(ats_score, 4),
                    market_competition_factor=1.0,
                ),
            )
            overall_ratio = breakdown.interview_chance_score

            out.append(
                {
                    **j,
                    "matched_jd_skills": scorecard.matched_jd_skills[:50],
                    "missing_jd_skills": scorecard.missing_jd_skills[:50],
                    "jd_alignment_percent": scorecard.jd_alignment_percent,
                    "missing_skills_gap_percent": scorecard.missing_skills_gap_percent,
                    "components": {
                        "skill_overlap": round(skill_overlap, 4),
                        "experience_alignment": round(semantic_exp_align, 4),
                        "ats_score": round(ats_score, 4),
                    },
                    "interview_chance_weights": weights.model_dump(),
                    "interview_chance_score": round(overall_ratio, 4),
                    "match_score": round(overall_ratio, 4),
                    "overall_match_percent": round(overall_ratio * 100.0, 2),
                }
            )

        top = max((float(j.get("match_score") or 0.0) for j in out), default=0.0)
        state.meta["l4_recursive_loop_required"] = top < 0.7
        state.meta["interview_chance_weights"] = weights.model_dump()
        if top < 0.7:
            state.log_eval(f"[L4] Recursive loop requested: top interview chance {top:.2f} < 0.70")
        return out
