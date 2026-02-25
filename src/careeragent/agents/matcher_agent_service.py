from __future__ import annotations

import re
from typing import Any, Dict, List

from careeragent.core.state import AgentState
from careeragent.nlp.skills import compute_jd_alignment, normalize_skill


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9+.#]", (text or "").lower()) if t and len(t) > 1]


class MatcherAgentService:
    """Description: Deterministic matcher (entity JD-alignment + experience heuristic).

    Layer: L4
    Input: extracted_profile + jobs_raw full text
    Output: jobs_scored list

    Notes:
      - JD Alignment is computed using entity/skill matching (NOT token-frequency).
      - This prevents ATS keyword match collapsing to tiny values (e.g., 0.05) due to noisy tokenization.
      - We persist both ratio and percent to avoid contract drift across UI / evaluators.
    """

    def score_jobs(self, state: AgentState) -> List[Dict[str, Any]]:
        prof = state.extracted_profile or {}
        resume_skills_raw = prof.get("skills") or []
        resume_skills = [normalize_skill(s) for s in resume_skills_raw if s]

        out: List[Dict[str, Any]] = []
        for j in state.jobs_raw:
            jd_text = (j.get("full_text_md") or j.get("snippet") or "")
            jd_title = str(j.get("title") or "")

            # --- Entity-based JD alignment (skills)
            scorecard = compute_jd_alignment(jd_text=jd_text, resume_skills=resume_skills)
            jd_align_ratio = scorecard.jd_alignment_percent / 100.0

            # --- Experience alignment heuristic (role keywords)
            toks = set(_tokenize(jd_text + " " + jd_title))
            exp_align = 0.25
            if any(k in toks for k in ["architect", "solution", "stakeholder", "roadmap", "strategy"]):
                exp_align = 0.65
            if any(k in toks for k in ["genai", "llm", "rag", "langchain", "langgraph", "azure", "openai"]):
                exp_align = min(1.0, exp_align + 0.2)
            if any(k in toks for k in ["mlops", "cicd", "docker", "kubernetes", "terraform", "mlflow"]):
                exp_align = min(1.0, exp_align + 0.15)

            # --- ATS proxy (for discovery-stage ranking only)
            # In L4 we can’t layout-check yet (that happens after generation).
            # We use JD alignment as a stable proxy for ATS keyword fit.
            ats_proxy = min(1.0, 0.05 + jd_align_ratio)

            # Weighted overall
            overall_ratio = 0.60 * jd_align_ratio + 0.25 * exp_align + 0.15 * ats_proxy
            overall_ratio = max(0.0, min(1.0, overall_ratio))

            out.append(
                {
                    **j,
                    "matched_jd_skills": scorecard.matched_jd_skills[:50],
                    "missing_jd_skills": scorecard.missing_jd_skills[:50],
                    "jd_alignment_percent": scorecard.jd_alignment_percent,
                    "missing_skills_gap_percent": scorecard.missing_skills_gap_percent,
                    "components": {
                        "jd_alignment": round(jd_align_ratio, 4),
                        "experience_alignment": round(exp_align, 4),
                        "ats_proxy": round(ats_proxy, 4),
                    },
                    # stable contract for downstream gating
                    "match_score": round(overall_ratio, 4),
                    "overall_match_percent": round(overall_ratio * 100.0, 2),
                }
            )

        return out
