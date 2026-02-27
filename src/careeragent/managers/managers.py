"""
GeoFence Manager  (L4)  +  Extraction Manager  (L5)
=====================================================
Both wrapped with SafeMethodResolver-compatible method aliases.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

log = logging.getLogger("managers")


# ══════════════════════════════════════════════════════════════════════════════
# GEO FENCE MANAGER  (L4)
# ══════════════════════════════════════════════════════════════════════════════

# US Metro aliases → lat/lon boxes  (extend as needed)
GEO_BOXES = {
    "new york":      {"lat": (40.4, 41.2), "lon": (-74.3, -73.7)},
    "san francisco": {"lat": (37.3, 38.0), "lon": (-122.6, -121.9)},
    "seattle":       {"lat": (47.3, 47.9), "lon": (-122.5, -121.9)},
    "austin":        {"lat": (30.0, 30.7), "lon": (-98.0, -97.4)},
    "boston":        {"lat": (42.2, 42.5), "lon": (-71.3, -70.9)},
    "chicago":       {"lat": (41.6, 42.1), "lon": (-87.9, -87.5)},
}

REMOTE_KEYWORDS = [
    "remote", "work from home", "wfh", "anywhere", "fully distributed",
    "100% remote", "distributed", "virtual",
]


class GeoFenceManager:
    """
    Canonical method: filter_by_geo(leads, geo_prefs) -> list[dict]

    Aliases: apply_geo_filter, geo_filter, filter
    """

    def filter_by_geo(self, leads: list[dict], geo_prefs: dict) -> list[dict]:
        if not leads:
            return []

        allow_remote    = geo_prefs.get("remote", True)
        target_locations = [l.lower() for l in geo_prefs.get("locations", [])]

        filtered = []
        for job in leads:
            if self._matches_geo(job, allow_remote, target_locations):
                filtered.append(job)

        log.info(
            "GeoFence: %d/%d leads passed (remote=%s, locations=%s)",
            len(filtered), len(leads), allow_remote, target_locations or "any",
        )
        return filtered

    # Aliases
    apply_geo_filter = filter_by_geo
    geo_filter       = filter_by_geo
    filter           = filter_by_geo

    # ── Internal ──────────────────────────────────────────────────────────────

    def _matches_geo(
        self,
        job: dict,
        allow_remote: bool,
        target_locations: list[str],
    ) -> bool:
        location_str = (job.get("location") or "").lower()
        is_remote    = (
            job.get("remote", False)
            or any(kw in location_str for kw in REMOTE_KEYWORDS)
        )

        # Accept remote jobs if remote preference is on
        if allow_remote and is_remote:
            return True

        # Accept if no specific location filter is set
        if not target_locations:
            return True

        # Accept if job location matches any target
        return any(loc in location_str for loc in target_locations)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION MANAGER  (L5)
# ══════════════════════════════════════════════════════════════════════════════

# Skills taxonomy for weighted matching
SKILL_WEIGHTS = {
    # Core language / framework match — high weight
    "python": 0.15, "javascript": 0.12, "typescript": 0.12, "java": 0.10,
    "react":  0.10, "node":       0.10, "fastapi":    0.08, "django": 0.08,
    "sql":    0.08, "postgresql":  0.07, "mongodb":   0.06,
    # Cloud / infra — medium weight
    "aws":  0.07, "gcp":    0.07, "azure":  0.07,
    "docker": 0.05, "kubernetes": 0.05, "terraform": 0.04,
    # Soft / process — low weight
    "agile": 0.02, "scrum": 0.02, "leadership": 0.02,
}


class ExtractionManager:
    """
    Canonical method: extract_and_score(leads, profile, threshold) -> list[dict]

    Aliases: score_jobs, extract_jd, score, run

    Scoring model:
      - Skill overlap (weighted bag-of-words)  → up to 0.60
      - Title similarity                        → up to 0.25
      - Seniority alignment                    → up to 0.15
    """

    def extract_and_score(
        self,
        leads: list[dict],
        profile: dict,
        threshold: float = 0.45,
    ) -> list[dict]:
        profile_skills = {s.lower() for s in profile.get("skills", [])}
        profile_title  = (profile.get("experience") or [{}])[0].get("title", "").lower()
        profile_years  = sum(
            e.get("years", 0) for e in profile.get("experience", [])
        )

        scored = []
        for job in leads:
            try:
                jd_text = (
                    job.get("description", "") + " " + job.get("title", "")
                ).lower()

                skill_score   = self._score_skills(jd_text, profile_skills)
                title_score   = self._score_title(job.get("title", "").lower(), profile_title)
                seniority_score = self._score_seniority(jd_text, profile_years)

                total = round(
                    0.60 * skill_score + 0.25 * title_score + 0.15 * seniority_score, 4
                )

                scored.append({
                    **job,
                    "score":           total,
                    "skill_score":     round(skill_score, 4),
                    "title_score":     round(title_score, 4),
                    "seniority_score": round(seniority_score, 4),
                    "matched_skills":  self._find_matches(jd_text, profile_skills),
                })
            except Exception as exc:
                log.warning("Scoring error for job '%s': %s", job.get("id"), exc)
                scored.append({**job, "score": 0.0})

        # Sort by score descending
        scored.sort(key=lambda j: j["score"], reverse=True)
        qualified = [j for j in scored if j["score"] >= threshold]
        log.info(
            "ExtractionManager: %d/%d jobs qualified at threshold=%.2f",
            len(qualified), len(scored), threshold,
        )
        # Return ALL scored (caller applies threshold)
        return scored

    # Aliases
    score_jobs  = extract_and_score
    extract_jd  = extract_and_score
    score       = extract_and_score
    run         = extract_and_score

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _score_skills(self, jd_text: str, profile_skills: set) -> float:
        if not profile_skills:
            return 0.0
        total_weight = 0.0
        matched_weight = 0.0
        for skill in profile_skills:
            w = SKILL_WEIGHTS.get(skill, 0.03)
            total_weight += w
            if skill in jd_text:
                matched_weight += w
        # Also scan JD for skills not in profile (penalty doesn't apply — only reward)
        return min(matched_weight / max(total_weight, 0.01), 1.0)

    def _score_title(self, jd_title: str, profile_title: str) -> float:
        if not profile_title or not jd_title:
            return 0.5   # neutral
        jd_words      = set(re.findall(r"\w+", jd_title))
        profile_words = set(re.findall(r"\w+", profile_title))
        overlap = jd_words & profile_words
        if not profile_words:
            return 0.5
        return min(len(overlap) / len(profile_words), 1.0)

    def _score_seniority(self, jd_text: str, profile_years: int) -> float:
        """Map years of experience to seniority tier and compare to JD requirements."""
        # Detect required years from JD
        yoe_matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)", jd_text)
        if not yoe_matches:
            return 0.7   # no explicit requirement → neutral-positive
        required = max(int(y) for y in yoe_matches)
        if profile_years >= required:
            return 1.0
        if profile_years >= required - 1:
            return 0.75
        if profile_years >= required - 2:
            return 0.5
        return 0.2

    def _find_matches(self, jd_text: str, profile_skills: set) -> list[str]:
        return sorted(s for s in profile_skills if s in jd_text)
