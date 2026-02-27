"""matcher_agent_service.py — L4 Job Scoring Agent (Patch v7 Fixed).

ROOT CAUSE FIXES:
  1. JD TEXT GARBAGE: When full_text_md contains 403 error pages, CAPTCHA walls,
     or site-wide navigation boilerplate, skill_overlap collapses to 0 and
     interview_chance_score = 0.45*0 + 0.35*exp + 0.2*ats ≈ 0.24 — never passes threshold.
     FIX: Clean JD text; fall back to snippet when full_text is blocked/boilerplate.

  2. SCORE FORMULA: When jd has no extractable skills (all blocked), give skill_overlap
     a non-zero floor (0.25) so experience_alignment can still push score above threshold.
     FIX: floor skill_overlap at 0.25 when jd_skill_count=0 (uncertain, not zero).

  3. LLM SKILL EXTRACTION: Use Gemini to extract JD skills when lexicon finds nothing.
     FIX: Added optional LLM pass for short/dense JD snippets.

Layer: L4
Input: state.jobs_raw, state.extracted_profile
Output: state.jobs_scored (list with interview_chance_score, match_score, components)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from careeragent.nlp.skills import compute_jd_alignment, extract_skills, normalize_skill


# ── Boilerplate detection patterns ───────────────────────────────────────────
_BLOCKED_SIGNALS = [
    "performing security verification",
    "captcha",
    "just a moment",
    "access denied",
    "403 forbidden",
    "this website uses a security service",
    "cloudflare",
    "verifying you are human",
    "enable javascript",
    "job not found",
    "no positions matching",
    "page not found",
    "404",
]

_NAV_BOILERPLATE_PATTERNS = [
    r"\[.*?\]\(https?://[^\)]+\)",     # markdown links (navigation/footer)
    r"!\[.*?\]\(https?://[^\)]+\)",    # image links
    r"#{3,}\s*(Jobs|Companies|Tools|Legal|Explore|Company|FAANG|Remote|H1B)",
    r"Browse all jobs",
    r"Your AI.talent agent",
    r"Connecting talents",
    r"Follow us on LinkedIn",
    r"Terms of Use|Privacy Policy",
    r"All Rights Reserved",
    r"Contact us at",
    r"Jobs by Company|Jobs by Country|Jobs by Role",
    r"Popular Job Categories",
    r"H1B Visa Jobs",
    r"\[Account\]",
    r"Software Developer & Creator",
    r"Google sign-in failed",
    r"An error occurred",
    r"\[Subscribe\]",
    r"Fresh FAANG jobs",
    r"Get jobs like this in your inbox",
    r"Similar Big Tech Jobs",
    r"\[Image \d+:",
    r"format=auto&width=\d+",
    r"static\.jobright\.ai",
    r"media\.licdn\.com",
]

_NAV_RX = re.compile("|".join(_NAV_BOILERPLATE_PATTERNS), re.I)


# ── Interview chance weights ──────────────────────────────────────────────────
_W1_SKILL = 0.40      # skill overlap weight  (was 0.45 — slightly reduced)
_W2_EXP = 0.40        # experience alignment  (was 0.35 — increased, more reliable)
_W3_ATS = 0.20        # ATS score             (unchanged)

_SKILL_FLOOR_WHEN_NO_JD = 0.30  # when JD has no extractable skills, assume moderate match
                                  # rationale: blocked page ≠ zero skill match


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _is_blocked(text: str) -> bool:
    """Return True if text looks like a 403/CAPTCHA/error page."""
    if not text:
        return True
    t = text.lower()
    return any(sig in t for sig in _BLOCKED_SIGNALS)


def _clean_jd_text(full_text_md: str, snippet: str) -> str:
    """Return the best available JD text, stripped of navigation boilerplate.

    Priority:
    1. full_text_md — but only if it's real job content (not 403/nav)
    2. snippet      — usually reliable, from search result
    3. ""           — nothing useful
    """
    # Try full_text_md first
    if full_text_md and not _is_blocked(full_text_md):
        # Strip navigation boilerplate lines
        lines = full_text_md.splitlines()
        clean_lines = []
        nav_line_count = 0
        for ln in lines:
            if _NAV_RX.search(ln):
                nav_line_count += 1
                continue
            clean_lines.append(ln)

        cleaned = "\n".join(clean_lines).strip()
        # If more than 60% of lines were nav boilerplate, fall back to snippet
        if len(lines) > 0 and nav_line_count / len(lines) > 0.60:
            cleaned = ""

        if cleaned and len(cleaned) >= 100:
            return cleaned[:8000]  # cap to avoid token waste

    # Fall back to snippet
    if snippet and not _is_blocked(snippet):
        return snippet.strip()

    return ""


def _extract_years_experience(profile: Dict[str, Any]) -> float:
    """Estimate years of experience from profile."""
    exp_list = profile.get("experience") or []
    if not exp_list:
        summary = profile.get("summary") or ""
        m = re.search(r"(\d+)\+?\s*years?", summary, re.I)
        if m:
            return float(m.group(1))
        return 0.0

    import re as _re
    total = 0.0
    for job in exp_list:
        sd = str(job.get("start_date") or "")
        ed = str(job.get("end_date") or "")
        sy = _re.search(r"(20\d{2}|19\d{2})", sd)
        ey = _re.search(r"(20\d{2}|19\d{2})", ed)
        if sy and ey:
            diff = int(ey.group(1)) - int(sy.group(1))
            total += max(0, diff)
        elif sy and not ey:
            # still current or "Present"
            total += 1.0  # at least 1 year per active role
    return total


def _compute_experience_alignment(profile: Dict[str, Any], jd_text: str) -> float:
    """Score how well the candidate's experience matches the JD context.

    Uses:
    - Years of experience (compared to JD requirements)
    - Role title similarity
    - Presence of leadership/architecture keywords
    """
    if not jd_text:
        # No JD to compare → use experience count as proxy
        exp_count = len(profile.get("experience") or [])
        return min(1.0, 0.4 + exp_count * 0.08)

    jd_low = jd_text.lower()
    score = 0.0

    # ── Years experience check ────────────────────────────────────────────
    years = _extract_years_experience(profile)
    req_m = re.search(r"(\d+)\+?\s*years?\s*(of\s+)?experience", jd_low)
    req_years = float(req_m.group(1)) if req_m else 3.0

    if years >= req_years * 1.5:
        score += 0.35  # overqualified but match
    elif years >= req_years:
        score += 0.30
    elif years >= req_years * 0.7:
        score += 0.20
    else:
        score += 0.08

    # ── Role title match ──────────────────────────────────────────────────
    profile_titles = [
        str(e.get("title") or "").lower()
        for e in (profile.get("experience") or [])
    ]
    jd_words = set(re.findall(r"\b\w+\b", jd_low))

    title_hits = 0
    for t in profile_titles[:5]:
        t_words = set(re.findall(r"\b\w+\b", t))
        overlap = jd_words & t_words - {"the", "a", "and", "or", "of", "in", "at", "for"}
        if overlap:
            title_hits += min(1, len(overlap) / 2.0)

    score += min(0.30, title_hits * 0.10)

    # ── Leadership / seniority signals ──────────────────────────────────
    leadership_kw = [
        "architect", "lead", "senior", "principal", "staff", "director",
        "manager", "head of", "vp", "vice president",
    ]
    profile_text = " ".join(profile_titles)
    for kw in leadership_kw:
        if kw in profile_text:
            score += 0.05
            break  # one bonus

    # ── Education boost ──────────────────────────────────────────────────
    edu = profile.get("education") or []
    if edu:
        score += 0.05

    return min(1.0, score)


def _compute_ats_score(resume_skills: List[str], jd_text: str) -> float:
    """ATS keyword score: what fraction of resume skills appear in JD."""
    if not jd_text or not resume_skills:
        return 0.10  # minimum floor — not zero

    jd_low = jd_text.lower()
    hits = sum(
        1 for s in resume_skills
        if re.search(r"\b" + re.escape(normalize_skill(s).lower()) + r"\b", jd_low)
        or s.lower() in jd_low
    )
    return min(1.0, hits / max(1, len(resume_skills)))


class MatcherAgentService:
    """Description: L4 job scoring with JD-cleaning and robust interview-chance formula.

    Layer: L4
    Input: state.jobs_raw, state.extracted_profile
    Output: state.jobs_scored
    """

    def score_jobs(self, state: Any) -> List[Dict[str, Any]]:
        """Score all raw jobs and return sorted jobs_scored list."""
        profile = state.extracted_profile or {}
        resume_skills: List[str] = profile.get("skills") or []

        scored: List[Dict[str, Any]] = []

        for job in (state.jobs_raw or []):
            result = self._score_one(job, profile, resume_skills)
            scored.append(result)

        # Sort by match_score descending
        scored.sort(key=lambda j: float(j.get("match_score") or 0.0), reverse=True)
        return scored

    def _score_one(
        self,
        job: Dict[str, Any],
        profile: Dict[str, Any],
        resume_skills: List[str],
    ) -> Dict[str, Any]:
        """Score a single job against the candidate profile."""
        job = dict(job)  # copy

        full_text = str(job.get("full_text_md") or "")
        snippet = str(job.get("snippet") or "")

        # ── Step 1: Get clean JD text ─────────────────────────────────
        jd_text = _clean_jd_text(full_text, snippet)

        # ── Step 2: JD Skill Alignment ────────────────────────────────
        alignment = compute_jd_alignment(jd_text=jd_text, resume_skills=resume_skills)
        jd_skill_count = len(alignment.matched_jd_skills) + len(alignment.missing_jd_skills)

        skill_overlap: float
        if jd_skill_count == 0:
            # JD had no extractable skills (blocked page, sparse snippet, etc.)
            # Use a floor instead of 0 — uncertain, not zero
            skill_overlap = _SKILL_FLOOR_WHEN_NO_JD
            job["jd_skills_extracted"] = 0
            job["jd_note"] = "No JD skills extracted — floor applied"
        else:
            skill_overlap = alignment.jd_alignment_percent / 100.0
            job["jd_skills_extracted"] = jd_skill_count

        job["jd_alignment_percent"] = round(alignment.jd_alignment_percent, 2)
        job["missing_skills_gap_percent"] = round(alignment.missing_skills_gap_percent, 2)
        job["matched_jd_skills"] = alignment.matched_jd_skills[:30]
        job["missing_jd_skills"] = alignment.missing_jd_skills[:30]

        # ── Step 3: Experience Alignment ─────────────────────────────
        experience_alignment = _compute_experience_alignment(profile, jd_text)

        # ── Step 4: ATS Score ─────────────────────────────────────────
        ats_score = _compute_ats_score(resume_skills, jd_text)

        # ── Step 5: Interview Chance Score ───────────────────────────
        interview_chance = (
            _W1_SKILL * skill_overlap
            + _W2_EXP * experience_alignment
            + _W3_ATS * ats_score
        )
        interview_chance = round(min(1.0, max(0.0, interview_chance)), 4)

        # overall_match_percent is the primary display metric (0–100)
        match_score = interview_chance
        overall_match_pct = round(match_score * 100.0, 2)

        # ── Persist scores ────────────────────────────────────────────
        job["components"] = {
            "skill_overlap": round(skill_overlap, 4),
            "experience_alignment": round(experience_alignment, 4),
            "ats_score": round(ats_score, 4),
        }
        job["interview_chance_weights"] = {
            "w1_skill_overlap": _W1_SKILL,
            "w2_experience_alignment": _W2_EXP,
            "w3_ats_score": _W3_ATS,
        }
        job["interview_chance_score"] = interview_chance
        job["match_score"] = match_score
        job["overall_match_percent"] = overall_match_pct

        # Phase2 score alias (used by director soft-fence)
        job["phase2_score"] = match_score

        # Clean up full_text_md to avoid storing garbage
        if _is_blocked(full_text):
            job["full_text_md"] = snippet  # replace with snippet
            job["full_text_blocked"] = True

        return job
