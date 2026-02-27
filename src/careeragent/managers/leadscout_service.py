"""
LeadScout Service — managers/leadscout_service.py
===================================================
THIS is the file main.py imports for L3 discovery.
Uses Serper /jobs endpoint with multi-query strategy.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional

log = logging.getLogger("leadscout")

SERPER_KEY  = os.getenv("SERPER_API_KEY", "")
TAVILY_KEY  = os.getenv("TAVILY_API_KEY", "")
REQUEST_TIMEOUT = 20


@dataclass
class JobLead:
    id:          str
    title:       str
    company:     str
    url:         str
    location:    str = ""
    remote:      bool = False
    salary_min:  Optional[int] = None
    salary_max:  Optional[int] = None
    description: str = ""
    source:      str = ""
    posted_date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LeadScoutService:
    """
    L3 Job discovery service.
    Called by main.py as:
        scout = LeadScoutService(enable_playwright_scrape=False)
        leads = await scout.search_jobs(intent)
    """

    def __init__(
        self,
        max_results_per_source: int = 25,
        enable_playwright_scrape: bool = False,
    ):
        self.max_per_source = max_results_per_source
        self.enable_playwright = enable_playwright_scrape

    # ── Entry point ─────────────────────────────────────────────────────────

    async def search_jobs(self, intent_plan: dict) -> list[dict]:
        queries  = self._build_queries(intent_plan)
        location = self._resolve_location(intent_plan)
        remote   = intent_plan.get("geo_preferences", {}).get("remote", True)

        log.info(
            "LeadScout starting: %d queries, location='%s'",
            len(queries), location,
        )
        for i, q in enumerate(queries):
            log.info("  Query[%d]: %s", i, q)

        tasks = []
        for query in queries:
            tasks.append(self._search_serper_jobs(query, location, remote))
            tasks.append(self._search_serper_organic(query, location))
            tasks.append(self._search_tavily(query, location, remote))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        leads: list[JobLead] = []
        for batch in results:
            if isinstance(batch, Exception):
                log.warning("LeadScout source error: %s", batch)
                continue
            if isinstance(batch, list):
                leads.extend(batch)

        # Deduplicate by URL
        seen, unique = set(), []
        for lead in leads:
            key = lead.url.strip().rstrip("/")
            if key and key not in seen:
                seen.add(key)
                unique.append(lead)

        log.info("LeadScout found %d unique leads (%d raw)", len(unique), len(leads))
        return [l.to_dict() for l in unique[:self.max_per_source * 4]]

    # Aliases
    find_jobs   = search_jobs
    scrape_jobs = search_jobs

    # ── Multi-query builder ──────────────────────────────────────────────────

    def _build_queries(self, intent_plan: dict) -> list[str]:
        roles    = intent_plan.get("target_roles", [])
        keywords = intent_plan.get("keywords", [])
        profile  = intent_plan.get("extracted_profile", {})

        # Pull all skills from profile
        profile_skills: list[str] = []
        if isinstance(profile.get("skills"), list):
            profile_skills = [str(s) for s in profile["skills"]]

        all_keywords = list(dict.fromkeys(keywords + profile_skills))

        # Bucket by domain
        ai_ml = [k for k in all_keywords if any(t in k.lower() for t in [
            "ai", "ml", "llm", "gpt", "bert", "transformer", "pytorch", "tensorflow",
            "langchain", "genai", "generative", "diffusion", "rag", "vector", "embedding",
            "hugging", "openai", "anthropic", "gemini", "llama", "fine-tun",
            "nlp", "computer vision", "deep learning", "neural", "langgraph",
        ])]
        cloud = [k for k in all_keywords if any(t in k.lower() for t in [
            "aws", "azure", "gcp", "sagemaker", "bedrock", "vertex", "cloud",
            "lambda", "kubernetes", "docker",
        ])]

        queries = []

        # Q1: top role + AI/ML core
        if roles:
            ai_str = " ".join(ai_ml[:4]) if ai_ml else " ".join(all_keywords[:4])
            queries.append(f"{roles[0]} {ai_str}")

        # Q2: seniority variant
        seniority = self._detect_seniority(profile, roles)
        if seniority and roles:
            queries.append(f"{seniority} {roles[0]}")

        # Q3: alt role titles
        for alt in self._alt_roles(roles)[:2]:
            ai_str = " ".join(ai_ml[:3]) if ai_ml else ""
            queries.append(f"{alt} {ai_str}".strip())

        # Q4: cloud combo
        if cloud and roles:
            queries.append(f"{roles[0]} {' '.join(cloud[:3])}")

        # Q5: GenAI specialist
        if any("gen" in k.lower() or "llm" in k.lower() for k in all_keywords):
            base = roles[0] if roles else "AI Engineer"
            queries.append(f"Generative AI {base} LLM")

        # Q6: ML fallback
        queries.append("Machine Learning Engineer GenAI remote")

        # Dedup, cap at 6
        seen_q: set[str] = set()
        final: list[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen_q:
                seen_q.add(q)
                final.append(q)
                if len(final) >= 6:
                    break

        return final or ["AI Engineer Python remote", "Machine Learning Engineer remote"]

    def _detect_seniority(self, profile: dict, roles: list[str]) -> str:
        combined = " ".join([
            str(profile.get("summary", "")),
            " ".join(str(r) for r in roles),
            " ".join(
                str(e.get("title", "") if isinstance(e, dict) else "")
                for e in profile.get("experience", [])
            ),
        ]).lower()
        if any(w in combined for w in ["principal", "staff", "distinguished", "vp", "director"]):
            return "Principal"
        if any(w in combined for w in ["senior", "sr.", "sr ", "lead", "architect"]):
            return "Senior"
        return ""

    def _alt_roles(self, roles: list[str]) -> list[str]:
        mapping = {
            "ai engineer":               ["Applied AI Engineer", "ML Engineer", "AI/ML Engineer"],
            "machine learning engineer": ["ML Engineer", "AI Engineer", "MLOps Engineer"],
            "data scientist":            ["Senior Data Scientist", "ML Researcher", "Applied Scientist"],
            "solution architect":        ["Solutions Architect AI", "Cloud AI Architect", "Technical Architect ML"],
            "genai":                     ["Generative AI Engineer", "LLM Engineer", "AI Engineer LLM"],
            "ai architect":              ["AI Solutions Architect", "ML Architect", "GenAI Architect"],
        }
        alts = []
        for role in roles:
            key = role.lower()
            for pattern, expansions in mapping.items():
                if pattern in key:
                    alts.extend(expansions)
        return list(dict.fromkeys(alts))

    def _resolve_location(self, intent_plan: dict) -> str:
        geo  = intent_plan.get("geo_preferences", {})
        locs = geo.get("locations", [])
        return locs[0] if locs else "United States"

    # ── Source: Serper /jobs ─────────────────────────────────────────────────

    async def _search_serper_jobs(self, query: str, location: str, remote: bool) -> list[JobLead]:
        if not SERPER_KEY:
            log.debug("Serper /jobs skipped — SERPER_API_KEY not set")
            return []
        try:
            import aiohttp
            search_q = f"{query} {'remote' if remote else location}"
            payload  = {"q": search_q, "gl": "us", "hl": "en", "num": self.max_per_source}
            headers  = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/jobs",
                    json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        log.warning("Serper /jobs HTTP %d for: %s", resp.status, query)
                        return []
                    data = await resp.json()

            leads = []
            for r in data.get("jobs", []):
                apply_link = ""
                opts = r.get("applyOptions", [])
                if opts and isinstance(opts, list):
                    apply_link = opts[0].get("link", "")
                url = apply_link or r.get("link", "")

                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", r.get("title", ""))[:40],
                    title       = r.get("title", ""),
                    company     = r.get("companyName", ""),
                    url         = url,
                    location    = r.get("location", ""),
                    remote      = "remote" in r.get("location", "").lower(),
                    description = r.get("description", "")[:500],
                    source      = "serper_jobs",
                    posted_date = (r.get("detectedExtensions") or {}).get("postedAt", ""),
                ))
            log.info("Serper /jobs: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Serper /jobs error for '%s': %s", query, exc)
            return []

    # ── Source: Serper /search (organic job board links) ────────────────────

    async def _search_serper_organic(self, query: str, location: str) -> list[JobLead]:
        if not SERPER_KEY:
            return []
        try:
            import aiohttp
            search_q = (
                f"{query} {location} "
                f"site:linkedin.com/jobs OR site:greenhouse.io OR site:lever.co OR site:indeed.com"
            )
            payload = {"q": search_q, "gl": "us", "num": 10}
            headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()

            leads = []
            for r in data.get("organic", []):
                url   = r.get("link", "")
                title = r.get("title", "")
                if not url or not title:
                    continue
                if not any(d in url for d in [
                    "linkedin.com/jobs", "greenhouse.io", "lever.co",
                    "indeed.com", "workday.com", "icims.com",
                ]):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = r.get("displayLink", ""),
                    url         = url,
                    description = r.get("snippet", "")[:400],
                    source      = "serper_organic",
                ))
            log.info("Serper organic: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Serper organic error: %s", exc)
            return []

    # ── Source: Tavily ───────────────────────────────────────────────────────

    async def _search_tavily(self, query: str, location: str, remote: bool) -> list[JobLead]:
        if not TAVILY_KEY:
            log.debug("Tavily skipped — TAVILY_API_KEY not set")
            return []
        try:
            import aiohttp
            loc_str = "remote" if remote else location
            payload = {
                "api_key":        TAVILY_KEY,
                "query":          f"{query} {loc_str} job opening",
                "search_depth":   "basic",
                "max_results":    self.max_per_source,
                "include_domains": [
                    "linkedin.com", "greenhouse.io", "lever.co",
                    "indeed.com", "workday.com", "icims.com",
                    "smartrecruiters.com", "jobvite.com",
                ],
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        log.warning("Tavily HTTP %d for: %s", resp.status, query)
                        return []
                    data = await resp.json()

            leads = []
            for r in data.get("results", []):
                url   = r.get("url", "")
                title = r.get("title", "")
                if not url or not title:
                    continue
                if any(skip in url for skip in ["/blog/", "/news/", "/about", "/company"]):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = "",
                    url         = url,
                    description = r.get("content", "")[:400],
                    source      = "tavily",
                ))
            log.info("Tavily: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Tavily error for '%s': %s", query, exc)
            return []
