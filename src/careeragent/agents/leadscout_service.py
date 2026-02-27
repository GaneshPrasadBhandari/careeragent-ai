"""
LeadScout Service  (L3)  — FIXED
==================================
Priority source chain (first working source wins per query):
  1. Serper.dev  — Google Jobs API  (uses SERPER_API_KEY from settings)
  2. Tavily      — Google Jobs search (uses TAVILY_API_KEY from settings)
  3. Adzuna      — Public job API (uses ADZUNA_APP_ID + ADZUNA_API_KEY)
  4. JSearch     — RapidAPI jobs   (uses JSEARCH_API_KEY)
  5. Playwright  — Direct ATS scraping (Greenhouse, Lever)

Key fix: runs MULTIPLE targeted queries from the full extracted profile
instead of a single weak `roles[0] + 3 keywords` query.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

log = logging.getLogger("leadscout")

# ── Config ─────────────────────────────────────────────────────────────────
SERPER_KEY     = os.getenv("SERPER_API_KEY", "")
TAVILY_KEY     = os.getenv("TAVILY_API_KEY", "")
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY", "")
JSEARCH_KEY    = os.getenv("JSEARCH_API_KEY", "")

REQUEST_TIMEOUT = 20
SCRAPE_TIMEOUT  = 30


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
    Canonical method: search_jobs(intent_plan: dict) -> list[dict]

    intent_plan shape expected (from L2 planner):
      {
        "target_roles":    ["Senior AI Engineer", "ML Engineer", ...],
        "keywords":        ["LangChain", "PyTorch", "AWS", "GenAI", ...],
        "geo_preferences": {"locations": ["United States"], "remote": True},
        "extracted_profile": { ... full profile dict from resume parser ... }
      }
    """

    def __init__(
        self,
        max_results_per_source: int = 25,
        enable_playwright_scrape: bool = True,
    ):
        self.max_per_source = max_results_per_source
        self.enable_playwright = enable_playwright_scrape

    # ── Canonical entry point ───────────────────────────────────────────────

    async def search_jobs(self, intent_plan: dict) -> list[dict]:
        queries  = self._build_queries(intent_plan)
        location = self._resolve_location(intent_plan)
        remote   = intent_plan.get("geo_preferences", {}).get("remote", True)

        log.info(
            "LeadScout starting search: %d queries, location='%s'",
            len(queries), location,
        )
        for i, q in enumerate(queries):
            log.debug("  Query[%d]: %s", i, q)

        # Run all queries across all sources concurrently
        tasks = []
        for query in queries:
            tasks.append(self._search_serper(query, location, remote))
            tasks.append(self._search_tavily(query, location, remote))
            tasks.append(self._search_adzuna(query, location))
            tasks.append(self._search_jsearch(query, location, remote))

        # Add ATS scraping for top roles
        top_roles = intent_plan.get("target_roles", [])[:3]
        if self.enable_playwright:
            for role in top_roles:
                tasks.append(self._scrape_greenhouse(role))
                tasks.append(self._scrape_lever(role))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        leads: list[JobLead] = []
        for batch in results:
            if isinstance(batch, Exception):
                log.warning("LeadScout source failed: %s", batch)
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

        log.info(
            "LeadScout found %d unique leads (%d raw)",
            len(unique), len(leads),
        )
        return [l.to_dict() for l in unique[: self.max_per_source * 4]]

    # Aliases
    find_jobs   = search_jobs
    scrape_jobs = search_jobs
    run         = search_jobs
    execute     = search_jobs

    # ── Query builder — THE KEY FIX ─────────────────────────────────────────

    def _build_queries(self, intent_plan: dict) -> list[str]:
        """
        Build multiple targeted queries instead of one weak generic query.
        Mines the full extracted_profile for roles, skills, and seniority.
        """
        roles    = intent_plan.get("target_roles", [])
        keywords = intent_plan.get("keywords", [])
        profile  = intent_plan.get("extracted_profile", {})

        # Pull additional context from the parsed resume if available
        profile_skills = []
        if isinstance(profile.get("skills"), list):
            profile_skills = [str(s) for s in profile["skills"]]
        elif isinstance(profile.get("skills"), dict):
            for v in profile["skills"].values():
                if isinstance(v, list):
                    profile_skills.extend([str(s) for s in v])

        # Merge keywords from intent_plan + profile skills
        all_keywords = list(dict.fromkeys(keywords + profile_skills))  # dedup, preserve order

        # Bucket skills by domain for targeted queries
        ai_ml_terms    = [k for k in all_keywords if any(t in k.lower() for t in [
            "ai", "ml", "llm", "gpt", "bert", "transformer", "pytorch", "tensorflow",
            "langchain", "genai", "generative", "diffusion", "rag", "vector", "embedding",
            "hugging", "openai", "anthropic", "gemini", "llama", "fine-tun",
            "nlp", "computer vision", "deep learning", "neural",
        ])]
        cloud_terms    = [k for k in all_keywords if any(t in k.lower() for t in [
            "aws", "azure", "gcp", "sagemaker", "bedrock", "vertex", "cloud",
            "lambda", "ec2", "s3", "kubernetes", "docker",
        ])]
        data_terms     = [k for k in all_keywords if any(t in k.lower() for t in [
            "python", "sql", "spark", "databricks", "snowflake", "pandas",
            "data science", "data engineer", "analytics", "bi",
        ])]

        queries = []

        # Query 1: Top role + AI/ML core skills
        if roles:
            core_ai = " ".join(ai_ml_terms[:4]) if ai_ml_terms else "LLM GenAI"
            queries.append(f"{roles[0]} {core_ai}")

        # Query 2: Senior/Staff variant if seniority detected in profile
        seniority = self._detect_seniority(profile, roles)
        if seniority and roles:
            queries.append(f"{seniority} {roles[0]}")

        # Query 3: Alt role titles for same persona
        alt_roles = self._alt_roles(roles)
        for alt in alt_roles[:2]:
            ai_slice = " ".join(ai_ml_terms[:3]) if ai_ml_terms else ""
            queries.append(f"{alt} {ai_slice}".strip())

        # Query 4: Cloud + AI combo
        if cloud_terms and roles:
            queries.append(f"{roles[0]} {' '.join(cloud_terms[:2])}")

        # Query 5: GenAI specialist query
        if any("gen" in k.lower() or "llm" in k.lower() for k in all_keywords):
            base = roles[0] if roles else "AI Engineer"
            queries.append(f"Generative AI {base} LLM")

        # Query 6: Data science / MLE fallback
        if data_terms:
            queries.append(f"Machine Learning Engineer {' '.join(data_terms[:3])}")

        # Deduplicate and cap
        seen_q: set[str] = set()
        final: list[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen_q:
                seen_q.add(q)
                final.append(q)
                if len(final) >= 6:
                    break

        # Fallback: never return empty
        if not final:
            final = ["AI Engineer Python remote", "Machine Learning Engineer remote"]

        return final

    def _detect_seniority(self, profile: dict, roles: list[str]) -> str:
        """Detect seniority level from profile experience."""
        combined = " ".join([
            str(profile.get("summary", "")),
            " ".join(str(r) for r in roles),
            " ".join(str(e.get("title", "")) for e in profile.get("experience", []) if isinstance(e, dict)),
        ]).lower()
        if any(w in combined for w in ["principal", "staff", "distinguished", "vp", "director"]):
            return "Principal"
        if any(w in combined for w in ["senior", "sr.", "sr ", "lead", "architect"]):
            return "Senior"
        return ""

    def _alt_roles(self, roles: list[str]) -> list[str]:
        """Expand role titles to common equivalents."""
        mapping = {
            "ai engineer":              ["Applied AI Engineer", "ML Engineer", "AI/ML Engineer"],
            "machine learning engineer":["ML Engineer", "AI Engineer", "MLOps Engineer"],
            "data scientist":           ["Senior Data Scientist", "ML Researcher", "Applied Scientist"],
            "solution architect":       ["Solutions Architect AI", "Cloud AI Architect", "Technical Architect ML"],
            "software engineer":        ["Backend Engineer", "Platform Engineer", "Senior Software Engineer"],
            "genai engineer":           ["Generative AI Engineer", "LLM Engineer", "AI Engineer LLM"],
        }
        alts = []
        for role in roles:
            key = role.lower().strip()
            for pattern, expansions in mapping.items():
                if pattern in key:
                    alts.extend(expansions)
        return list(dict.fromkeys(alts))  # dedup

    def _resolve_location(self, intent_plan: dict) -> str:
        geo = intent_plan.get("geo_preferences", {})
        locs = geo.get("locations", [])
        if locs:
            return locs[0]
        return "United States"

    # ── Source: Serper.dev Google Jobs ──────────────────────────────────────

    async def _search_serper(self, query: str, location: str, remote: bool) -> list[JobLead]:
        """Google Jobs via Serper.dev — most reliable free source."""
        if not SERPER_KEY:
            log.debug("Serper skipped — SERPER_API_KEY not set.")
            return []
        try:
            import aiohttp
            search_q = f"{query} {'remote' if remote else location}"
            payload = {
                "q": search_q,
                "gl": "us",
                "hl": "en",
                "num": self.max_per_source,
            }
            headers = {
                "X-API-KEY":    SERPER_KEY,
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/jobs",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        log.warning("Serper HTTP %d for query: %s", resp.status, query)
                        return []
                    data = await resp.json()

            leads = []
            for r in data.get("jobs", []):
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", r.get("title", ""))[:40],
                    title       = r.get("title", ""),
                    company     = r.get("company_name", ""),
                    url         = r.get("job_highlights", {}).get("apply_link", "") or r.get("detected_extensions", {}).get("job_id", ""),
                    location    = r.get("location", ""),
                    remote      = "remote" in r.get("location", "").lower(),
                    description = " ".join(r.get("highlights", {}).get("Qualifications", [])),
                    source      = "serper_jobs",
                    posted_date = r.get("detected_extensions", {}).get("posted_at", ""),
                ))
            log.info("Serper returned %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Serper error for query '%s': %s", query, exc)
            return []

    # ── Source: Serper.dev Google Jobs (organic fallback) ───────────────────

    async def _search_serper_organic(self, query: str, location: str) -> list[JobLead]:
        """Fallback: Serper organic search for job listings."""
        if not SERPER_KEY:
            return []
        try:
            import aiohttp
            search_q = f"{query} {location} site:linkedin.com/jobs OR site:greenhouse.io OR site:lever.co OR site:indeed.com"
            payload  = {"q": search_q, "gl": "us", "num": 10}
            headers  = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
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
                # Filter to known job board URLs
                if not any(d in url for d in ["linkedin.com/jobs", "greenhouse.io", "lever.co", "indeed.com", "workday.com"]):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = r.get("displayLink", ""),
                    url         = url,
                    description = r.get("snippet", ""),
                    source      = "serper_organic",
                ))
            log.info("Serper organic returned %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Serper organic error: %s", exc)
            return []

    # ── Source: Tavily ──────────────────────────────────────────────────────

    async def _search_tavily(self, query: str, location: str, remote: bool) -> list[JobLead]:
        """Tavily search for job listings across job boards."""
        if not TAVILY_KEY:
            log.debug("Tavily skipped — TAVILY_API_KEY not set.")
            return []
        try:
            import aiohttp
            loc_str  = "remote" if remote else location
            full_q   = (
                f"{query} {loc_str} job opening 2024 2025 "
                f"site:linkedin.com OR site:greenhouse.io OR site:lever.co OR site:indeed.com"
            )
            payload  = {
                "api_key":        TAVILY_KEY,
                "query":          full_q,
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
                        log.warning("Tavily HTTP %d for query: %s", resp.status, query)
                        return []
                    data = await resp.json()

            leads = []
            for r in data.get("results", []):
                url   = r.get("url", "")
                title = r.get("title", "")
                if not url or not title:
                    continue
                # Skip non-job pages
                if any(skip in url for skip in ["/blog/", "/news/", "/about", "/company"]):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = "",
                    url         = url,
                    description = r.get("content", "")[:500],
                    source      = "tavily",
                ))
            log.info("Tavily returned %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Tavily error for query '%s': %s", query, exc)
            return []

    # ── Source: Adzuna API ──────────────────────────────────────────────────

    async def _search_adzuna(self, query: str, location: str) -> list[JobLead]:
        if not ADZUNA_APP_ID or not ADZUNA_API_KEY:
            log.debug("Adzuna skipped — API keys not set.")
            return []
        try:
            import aiohttp
            from urllib.parse import quote_plus
            url = (
                f"https://api.adzuna.com/v1/api/jobs/us/search/1"
                f"?app_id={ADZUNA_APP_ID}&app_key={ADZUNA_API_KEY}"
                f"&results_per_page={self.max_per_source}"
                f"&what={quote_plus(query)}&where={quote_plus(location)}&content-type=application/json"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                    if resp.status != 200:
                        log.warning("Adzuna HTTP %d", resp.status)
                        return []
                    data = await resp.json()
            leads = []
            for r in data.get("results", []):
                leads.append(JobLead(
                    id          = str(r.get("id", "")),
                    title       = r.get("title", ""),
                    company     = r.get("company", {}).get("display_name", ""),
                    url         = r.get("redirect_url", ""),
                    location    = r.get("location", {}).get("display_name", ""),
                    salary_min  = r.get("salary_min"),
                    salary_max  = r.get("salary_max"),
                    description = r.get("description", ""),
                    source      = "adzuna",
                    posted_date = r.get("created", ""),
                ))
            log.info("Adzuna returned %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Adzuna error: %s", exc)
            return []

    # ── Source: JSearch (RapidAPI) ──────────────────────────────────────────

    async def _search_jsearch(self, query: str, location: str, remote: bool) -> list[JobLead]:
        if not JSEARCH_KEY:
            log.debug("JSearch skipped — JSEARCH_API_KEY not set.")
            return []
        try:
            import aiohttp
            params = {
                "query":            f"{query} {location}",
                "page":             "1",
                "num_pages":        "2",
                "date_posted":      "week",
                "employment_types": "FULLTIME",
            }
            if remote:
                params["remote_jobs_only"] = "true"
            headers = {
                "X-RapidAPI-Key":  JSEARCH_KEY,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://jsearch.p.rapidapi.com/search",
                    params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        log.warning("JSearch HTTP %d", resp.status)
                        return []
                    data = await resp.json()
            leads = []
            for r in data.get("data", []):
                leads.append(JobLead(
                    id          = r.get("job_id", ""),
                    title       = r.get("job_title", ""),
                    company     = r.get("employer_name", ""),
                    url         = r.get("job_apply_link") or r.get("job_google_link", ""),
                    location    = f"{r.get('job_city','')}, {r.get('job_country','')}".strip(", "),
                    remote      = r.get("job_is_remote", False),
                    salary_min  = r.get("job_min_salary"),
                    salary_max  = r.get("job_max_salary"),
                    description = r.get("job_description", ""),
                    source      = "jsearch",
                    posted_date = r.get("job_posted_at_datetime_utc", ""),
                ))
            log.info("JSearch returned %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("JSearch error: %s", exc)
            return []

    # ── Source: Playwright → Greenhouse ────────────────────────────────────

    async def _scrape_greenhouse(self, role: str) -> list[JobLead]:
        if not self.enable_playwright:
            return []
        try:
            search_url = f"https://boards.greenhouse.io/search#t={role.replace(' ', '+')}"
            return await self._playwright_scrape(search_url, "greenhouse", role)
        except Exception as exc:
            log.warning("Greenhouse scrape failed: %s", exc)
            return []

    # ── Source: Playwright → Lever ──────────────────────────────────────────

    async def _scrape_lever(self, role: str) -> list[JobLead]:
        if not self.enable_playwright:
            return []
        try:
            search_url = (
                f"https://www.google.com/search?q=site:jobs.lever.co+{role.replace(' ', '+')}"
            )
            return await self._playwright_scrape(search_url, "lever", role)
        except Exception as exc:
            log.warning("Lever scrape failed: %s", exc)
            return []

    async def _playwright_scrape(self, url: str, source: str, role: str) -> list[JobLead]:
        try:
            from playwright.async_api import async_playwright, TimeoutError as PWTimeout
        except ImportError:
            log.debug("Playwright not installed — skipping %s scrape", source)
            return []

        leads = []
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
                page    = await browser.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=SCRAPE_TIMEOUT * 1000)
                    await page.wait_for_timeout(2000)

                    if source == "greenhouse":
                        items = await page.query_selector_all(".opening")
                        for item in items[:self.max_per_source]:
                            link = await item.query_selector("a")
                            if not link:
                                continue
                            title = await link.inner_text()
                            href  = await link.get_attribute("href")
                            if href:
                                leads.append(JobLead(
                                    id=re.sub(r"\W+", "_", title)[:40],
                                    title=title.strip(),
                                    company="",
                                    url=href,
                                    source=source,
                                ))
                    elif source == "lever":
                        links = await page.query_selector_all("a[href*='jobs.lever.co']")
                        for link in links[:self.max_per_source]:
                            href  = await link.get_attribute("href")
                            title = await link.inner_text()
                            if href and "jobs.lever.co" in href:
                                leads.append(JobLead(
                                    id=re.sub(r"\W+", "_", title)[:40],
                                    title=title.strip(),
                                    company="",
                                    url=href,
                                    source=source,
                                ))
                except PWTimeout:
                    log.warning("Playwright timeout scraping %s", url)
                finally:
                    await browser.close()
        except Exception as exc:
            log.warning("Playwright %s scrape error: %s", source, exc)

        log.info("%s scrape returned %d leads for role: %s", source, len(leads), role)
        return leads
