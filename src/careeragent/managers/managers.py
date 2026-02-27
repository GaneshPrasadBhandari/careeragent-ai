"""
GeoFence Manager  (L4)  +  Extraction Manager  (L5)
=====================================================
Fixed: SKILL_WEIGHTS now includes full AI/ML/GenAI/Cloud taxonomy.
Fixed: Scoring uses description + title + snippet for better coverage.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("managers")


# ══════════════════════════════════════════════════════════════════════════════
# GEO FENCE MANAGER  (L4)
# ══════════════════════════════════════════════════════════════════════════════

REMOTE_KEYWORDS = [
    "remote", "work from home", "wfh", "anywhere", "fully distributed",
    "100% remote", "distributed", "virtual", "telecommute",
]


class GeoFenceManager:
    """
    Canonical method: filter_by_geo(leads, geo_prefs) -> list[dict]
    Aliases: apply_geo_filter, geo_filter, filter
    """

    def filter_by_geo(self, leads: list[dict], geo_prefs: dict) -> list[dict]:
        if not leads:
            return []

        allow_remote     = geo_prefs.get("remote", True)
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

    apply_geo_filter = filter_by_geo
    geo_filter       = filter_by_geo
    filter           = filter_by_geo

    def _matches_geo(self, job: dict, allow_remote: bool, target_locations: list[str]) -> bool:
        location_str = (job.get("location") or "").lower()
        desc_str     = (job.get("description") or job.get("snippet") or "").lower()
        combined     = location_str + " " + desc_str

        is_remote = (
            job.get("remote", False)
            or any(kw in combined for kw in REMOTE_KEYWORDS)
        )

        if allow_remote and is_remote:
            return True
        if not target_locations:
            return True
        return any(loc in location_str for loc in target_locations)


# ══════════════════════════════════════════════════════════════════════════════
# SKILL WEIGHTS  — full taxonomy
# ══════════════════════════════════════════════════════════════════════════════

SKILL_WEIGHTS: Dict[str, float] = {
    # ── AI / ML / GenAI — HIGH weight (these are the target skills) ──────────
    "machine learning":      0.14,
    "deep learning":         0.14,
    "generative ai":         0.14,
    "genai":                 0.13,
    "llm":                   0.13,
    "large language model":  0.13,
    "llms":                  0.12,
    "gpt":                   0.11,
    "transformer":           0.11,
    "pytorch":               0.12,
    "tensorflow":            0.12,
    "langchain":             0.12,
    "langgraph":             0.12,
    "rag":                   0.12,
    "retrieval augmented":   0.11,
    "vector database":       0.11,
    "vector db":             0.11,
    "embeddings":            0.10,
    "embedding":             0.10,
    "fine-tuning":           0.11,
    "fine tuning":           0.11,
    "hugging face":          0.10,
    "huggingface":           0.10,
    "nlp":                   0.11,
    "natural language":      0.10,
    "computer vision":       0.10,
    "reinforcement learning":0.10,
    "rlhf":                  0.11,
    "prompt engineering":    0.10,
    "openai":                0.10,
    "anthropic":             0.10,
    "gemini":                0.09,
    "llama":                 0.09,
    "mistral":               0.09,
    "bert":                  0.09,
    "diffusion":             0.09,
    "agentic":               0.10,
    "multi-agent":           0.10,
    "mlops":                 0.11,
    "model deployment":      0.10,
    "model serving":         0.10,
    "ai engineer":           0.12,
    "ml engineer":           0.12,
    "data scientist":        0.11,
    "applied scientist":     0.11,
    "research scientist":    0.11,

    # ── Cloud & Infrastructure ────────────────────────────────────────────────
    "aws":          0.10,
    "amazon web services": 0.10,
    "sagemaker":    0.11,
    "bedrock":      0.11,
    "azure":        0.09,
    "gcp":          0.09,
    "google cloud": 0.09,
    "vertex ai":    0.10,
    "kubernetes":   0.08,
    "docker":       0.07,
    "terraform":    0.06,
    "lambda":       0.06,
    "ec2":          0.05,
    "s3":           0.05,
    "cloud":        0.06,

    # ── Core languages / frameworks ───────────────────────────────────────────
    "python":       0.12,
    "sql":          0.08,
    "spark":        0.08,
    "databricks":   0.08,
    "snowflake":    0.07,
    "kafka":        0.06,
    "airflow":      0.07,
    "fastapi":      0.06,
    "django":       0.05,
    "flask":        0.05,
    "javascript":   0.05,
    "typescript":   0.05,
    "java":         0.05,
    "scala":        0.06,
    "r":            0.05,

    # ── Data & Analytics ─────────────────────────────────────────────────────
    "pandas":       0.06,
    "numpy":        0.06,
    "scikit-learn": 0.08,
    "sklearn":      0.07,
    "mlflow":       0.08,
    "dvc":          0.07,
    "feature engineering": 0.07,
    "data engineering":    0.07,
    "data science":        0.09,
    "analytics":           0.05,

    # ── Databases ────────────────────────────────────────────────────────────
    "postgresql": 0.05, "mysql": 0.04, "mongodb": 0.05,
    "redis": 0.05, "elasticsearch": 0.05, "pinecone": 0.07,
    "chroma": 0.06, "faiss": 0.07, "weaviate": 0.06, "qdrant": 0.06,

    # ── Architecture ─────────────────────────────────────────────────────────
    "solution architect":   0.10,
    "solutions architect":  0.10,
    "technical architect":  0.09,
    "system design":        0.07,
    "microservices":        0.06,
    "rest":                 0.04,
    "graphql":              0.04,
    "grpc":                 0.05,

    # ── Process / soft ───────────────────────────────────────────────────────
    "agile": 0.02, "scrum": 0.02, "leadership": 0.03,
    "communication": 0.02, "cross-functional": 0.02,
}


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION MANAGER  (L5)
# ══════════════════════════════════════════════════════════════════════════════

class ExtractionManager:
    """
    Canonical method: extract_and_score(leads, profile, threshold) -> list[dict]
    Aliases: score_jobs, extract_jd, score, run

    Scoring model:
      skill_score   (0.60) — weighted skill overlap using full AI/ML taxonomy
      title_score   (0.25) — title word overlap
      seniority_score (0.15) — years of experience vs JD requirement
    """

    def extract_and_score(
        self,
        leads: list[dict],
        profile: dict,
        threshold: float = 0.45,
    ) -> list[dict]:
        # Build profile skill set — normalise to lowercase
        profile_skills: Set[str] = set()
        for s in profile.get("skills", []):
            profile_skills.add(str(s).lower().strip())

        # Also extract skills from raw_text if available (catches skills
        # the regex parser missed)
        raw_text = (profile.get("raw_text") or "").lower()
        for skill_key in SKILL_WEIGHTS:
            if skill_key in raw_text:
                profile_skills.add(skill_key)

        profile_title = ""
        exp = profile.get("experience") or []
        if exp:
            first = exp[0]
            profile_title = (
                first.get("title", "") if isinstance(first, dict) else ""
            ).lower()

        profile_years = sum(
            (e.get("years", 0) if isinstance(e, dict) else 0)
            for e in exp
        )

        log.info(
            "ExtractionManager: scoring %d leads, profile_skills=%d, threshold=%.2f",
            len(leads), len(profile_skills), threshold,
        )

        scored = []
        for job in leads:
            try:
                # Combine all text fields for matching
                jd_text = " ".join([
                    job.get("description", ""),
                    job.get("title", ""),
                    job.get("snippet", ""),
                    job.get("company", ""),
                ]).lower()

                skill_score     = self._score_skills(jd_text, profile_skills)
                title_score     = self._score_title(job.get("title", "").lower(), profile_title)
                seniority_score = self._score_seniority(jd_text, profile_years)

                total = round(
                    0.60 * skill_score
                    + 0.25 * title_score
                    + 0.15 * seniority_score,
                    4,
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

        scored.sort(key=lambda j: j["score"], reverse=True)
        qualified = [j for j in scored if j["score"] >= threshold]
        log.info(
            "ExtractionManager: %d/%d jobs qualified at threshold=%.2f  (top=%.3f)",
            len(qualified), len(scored), threshold,
            scored[0]["score"] if scored else 0.0,
        )
        return scored  # caller applies threshold filter

    # Aliases
    score_jobs = extract_and_score
    extract_jd = extract_and_score
    score      = extract_and_score
    run        = extract_and_score

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _score_skills(self, jd_text: str, profile_skills: Set[str]) -> float:
        if not profile_skills:
            return 0.3  # neutral rather than 0

        matched_weight = 0.0
        total_weight   = 0.0

        for skill in profile_skills:
            w = SKILL_WEIGHTS.get(skill, 0.04)
            total_weight += w
            if skill in jd_text:
                matched_weight += w

        if total_weight == 0:
            return 0.3

        raw = matched_weight / total_weight

        # Boost: if JD text is short (snippet-only) we can't expect many keyword
        # hits — apply a generous floor so short-description jobs aren't auto-failed
        if len(jd_text) < 300:
            raw = max(raw, 0.35)

        return min(raw, 1.0)

    def _score_title(self, jd_title: str, profile_title: str) -> float:
        if not jd_title:
            return 0.5

        # Broad title family matching — handles AI/ML/GenAI variants
        ai_title_keywords = {
            "ai", "ml", "machine learning", "deep learning", "data scientist",
            "data science", "llm", "genai", "generative", "nlp", "computer vision",
            "research", "applied", "architect", "engineer", "scientist",
        }
        jd_words      = set(re.findall(r"\w+", jd_title))
        profile_words = set(re.findall(r"\w+", profile_title)) if profile_title else set()

        # Direct word overlap
        direct_overlap = len(jd_words & profile_words) / max(len(profile_words), 1) if profile_words else 0.0

        # AI/ML family boost — if both titles are in the same domain family
        jd_in_ai      = bool(jd_words & ai_title_keywords)
        profile_in_ai = bool(profile_words & ai_title_keywords) if profile_words else False

        if jd_in_ai and (profile_in_ai or not profile_title):
            return max(direct_overlap, 0.70)  # strong boost for AI-to-AI matching

        return max(direct_overlap, 0.5)  # neutral floor

    def _score_seniority(self, jd_text: str, profile_years: int) -> float:
        yoe_matches = re.findall(
            r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)", jd_text
        )
        if not yoe_matches:
            return 0.75  # no explicit requirement → positive default

        required = max(int(y) for y in yoe_matches)
        if profile_years >= required:
            return 1.0
        if profile_years >= required - 2:
            return 0.75
        if profile_years >= required - 4:
            return 0.55
        return 0.30

    def _find_matches(self, jd_text: str, profile_skills: Set[str]) -> List[str]:
        return sorted(s for s in profile_skills if s in jd_text)[:20]
