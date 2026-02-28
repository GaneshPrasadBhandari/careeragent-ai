from __future__ import annotations

import re
from typing import Any, Dict, List

from careeragent.nlp.skills import compute_jd_alignment, extract_skills, normalize_skill

_W_SKILL = 0.45
_W_EXP = 0.35
_W_ATS = 0.20

_BLOCKED_SIGNALS = (
    "captcha",
    "access denied",
    "403",
    "404",
    "cloudflare",
    "verify you are human",
    "page not found",
)


def _clean_jd_text(full_text_md: str, snippet: str) -> str:
    full = (full_text_md or "").strip()
    snip = (snippet or "").strip()
    if full and not any(x in full.lower() for x in _BLOCKED_SIGNALS):
        return full[:12000]
    if snip and not any(x in snip.lower() for x in _BLOCKED_SIGNALS):
        return snip[:4000]
    return ""


def _extract_years(profile: Dict[str, Any]) -> float:
    years = 0.0
    for exp in (profile.get("experience") or []):
        blob = " ".join(str(exp.get(k) or "") for k in ("title", "start_date", "end_date"))
        matches = re.findall(r"(19\d{2}|20\d{2})", blob)
        if len(matches) >= 2:
            ys = sorted(int(x) for x in matches)
            years = max(years, float(max(0, ys[-1] - ys[0])))
    if years > 0:
        return min(25.0, years)
    # pragmatic fallback when dates are missing
    return min(20.0, float(len(profile.get("experience") or [])) * 1.8)


def _experience_alignment(profile: Dict[str, Any], jd_text: str) -> float:
    years = _extract_years(profile)
    jd_low = (jd_text or "").lower()
    req = 3.0
    m = re.search(r"(\d{1,2})\+?\s*years", jd_low)
    if m:
        req = float(m.group(1))
    ratio = min(1.0, years / max(1.0, req))

    bonus = 0.0
    role_blob = " ".join(str(e.get("title") or "") for e in (profile.get("experience") or []))
    role_low = role_blob.lower()
    if any(k in role_low for k in ["architect", "lead", "principal", "manager"]):
        bonus += 0.15
    if any(k in jd_low for k in ["architect", "lead", "principal"]) and bonus == 0:
        bonus += 0.05
    return min(1.0, ratio + bonus)


def _ats_keyword_match(resume_skills: List[str], jd_text: str) -> float:
    if not jd_text:
        return 0.2
    jd_skills = set(extract_skills(jd_text, extra_candidates=resume_skills))
    if not jd_skills:
        return 0.35
    rs = {normalize_skill(s) for s in resume_skills if s}
    matched = len(jd_skills & rs)
    return max(0.1, min(1.0, matched / max(1, len(jd_skills))))


class MatcherAgentService:
    """L4 scorer with robust fallback metrics and HITL-facing explanations."""

    def score_jobs(self, state: Any) -> List[Dict[str, Any]]:
        profile = state.extracted_profile or {}
        resume_skills = [normalize_skill(s) for s in (profile.get("skills") or []) if s]
        scored: List[Dict[str, Any]] = []

        for job in (state.jobs_raw or []):
            jd_text = _clean_jd_text(str(job.get("full_text_md") or ""), str(job.get("snippet") or ""))
            align = compute_jd_alignment(jd_text=jd_text, resume_skills=resume_skills)

            skill_overlap = (align.jd_alignment_percent / 100.0) if (align.jd_alignment_percent > 0) else 0.30
            exp_align = _experience_alignment(profile, jd_text)
            ats_kw = _ats_keyword_match(resume_skills, jd_text)

            interview = (_W_SKILL * skill_overlap) + (_W_EXP * exp_align) + (_W_ATS * ats_kw)
            interview = max(0.0, min(1.0, interview))
            offer = max(0.0, min(1.0, interview * 0.78))

            enriched = {
                **dict(job),
                "matched_jd_skills": align.matched_jd_skills[:40],
                "missing_jd_skills": align.missing_jd_skills[:40],
                "jd_alignment_percent": round(align.jd_alignment_percent, 2),
                "missing_skills_gap_percent": round(align.missing_skills_gap_percent, 2),
                "ats_keyword_match_percent": round(ats_kw * 100.0, 2),
                "interview_probability_percent": round(interview * 100.0, 2),
                "job_offer_probability_percent": round(offer * 100.0, 2),
                "interview_chance_score": round(interview, 4),
                "match_score": round(interview, 4),
                "overall_match_percent": round(interview * 100.0, 2),
                "components": {
                    "skill_overlap": round(skill_overlap, 4),
                    "experience_alignment": round(exp_align, 4),
                    "ats_score": round(ats_kw, 4),
                    "ats_proxy": round(ats_kw, 4),
                },
                "selection_reason": (
                    f"Skill match {round(align.jd_alignment_percent,1)}%, experience fit {round(exp_align*100,1)}%, "
                    f"ATS keyword fit {round(ats_kw*100,1)}%."
                ),
            }
            scored.append(enriched)

        scored.sort(key=lambda x: float(x.get("match_score") or 0.0), reverse=True)
        top = max((float(x.get("match_score") or 0.0) for x in scored), default=0.0)
        state.meta["l4_recursive_loop_required"] = top < 0.7
        return scored
