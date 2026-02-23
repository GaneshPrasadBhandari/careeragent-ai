from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

from careeragent.core.state import AgentState


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9+.#]", (text or "").lower()) if t and len(t) > 1]


def _keyword_set(text: str) -> set:
    return set(_tokenize(text))


class MatcherAgentService:
    """Description: Lightweight matcher (skill overlap + ATS keyword score).
    Layer: L4
    Input: extracted_profile + jobs_raw full text
    Output: jobs_scored list
    """

    def score_jobs(self, state: AgentState) -> List[Dict[str, Any]]:
        prof = state.extracted_profile or {}
        skills = prof.get("skills") or []
        skill_set = set([s.lower() for s in skills])

        out: List[Dict[str, Any]] = []
        for j in state.jobs_raw:
            text = (j.get("full_text_md") or j.get("snippet") or "")
            tokens = _keyword_set(text)

            # skill overlap: how many resume skills appear
            matched = [s for s in skills if s.lower() in tokens]
            missing = [s for s in skills[:20] if s.lower() not in tokens]  # top slice only

            skill_overlap = len(matched) / max(1, min(len(skills), 40))

            # experience alignment heuristic: presence of "architect" etc.
            exp_align = 0.2
            if any(k in tokens for k in ["architect", "solution", "stakeholder", "roadmap"]):
                exp_align = 0.6
            if any(k in tokens for k in ["genai", "llm", "rag", "langchain", "langgraph", "azure"]):
                exp_align = min(1.0, exp_align + 0.25)

            # ATS score: overlap between job keywords and resume keywords
            resume_kw = set([s.lower() for s in skills]) | _keyword_set(prof.get("summary") or "")
            ats = len(tokens & resume_kw) / max(1, len(tokens & set(list(tokens)[:250]) ) )
            ats = max(0.0, min(1.0, ats))

            overall = 0.55 * skill_overlap + 0.25 * exp_align + 0.20 * ats

            out.append(
                {
                    **j,
                    "matched_skills": matched[:40],
                    "missing_skills": missing[:20],
                    "components": {"skill_overlap": skill_overlap, "experience_alignment": exp_align, "ats_score": ats},
                    "overall_match_percent": round(overall * 100.0, 2),
                }
            )

        return out
