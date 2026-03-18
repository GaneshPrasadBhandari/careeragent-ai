"""
LeadScout Service — managers/leadscout_service.py
===================================================
Uses httpx (already in deps) instead of aiohttp.
Serper /jobs removed — returns 404 on this plan.
Uses Serper /search organic + Tavily as primary sources.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urlencode, urlsplit, urlunsplit

import httpx

log = logging.getLogger("leadscout")

SERPER_KEY      = os.getenv("SERPER_API_KEY", "")
TAVILY_KEY      = os.getenv("TAVILY_API_KEY", "")
REQUEST_TIMEOUT = 20.0

JOB_BOARD_DOMAINS = [
    "linkedin.com/jobs",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "naukri.com",
    "wellfound.com",
    "monsterindia.com",
    "greenhouse.io",
    "lever.co",
    "workday.com",
    "myworkdayjobs.com",
    "icims.com",
    "smartrecruiters.com",
    "jobvite.com",
    "ashbyhq.com",
    "rippling.com",
]

COUNTRY_SOURCE_ROTATION = {
    "US": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
        {"label": "ZipRecruiter", "domains": ["ziprecruiter.com"]},
    ],
    "IN": [
        {"label": "Naukri", "domains": ["naukri.com"]},
        {"label": "Wellfound", "domains": ["wellfound.com", "angel.co"]},
        {"label": "Monster", "domains": ["monsterindia.com", "foundit.in"]},
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
    ],
    "EU": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
    ],
    "AU": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
    ],
    "UAE": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
    ],
}

COUNTRY_SEARCH_PRESETS = {
    "US": {"code": "US", "label": "United States", "location": "United States", "gl": "us", "hl": "en"},
    "IN": {"code": "IN", "label": "India", "location": "India", "gl": "in", "hl": "en"},
    "EU": {"code": "EU", "label": "Europe", "location": "Europe", "gl": "de", "hl": "en"},
    "AU": {"code": "AU", "label": "Australia", "location": "Australia", "gl": "au", "hl": "en"},
    "UAE": {"code": "UAE", "label": "UAE", "location": "United Arab Emirates", "gl": "ae", "hl": "en"},
}

SKIP_PATHS = ["/blog/", "/news/", "/about", "/company", "/press", "/learn"]


def sanitize_job_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = f"https:{raw}"
    if not raw.startswith(("http://", "https://")):
        return raw

    try:
        parts = urlsplit(raw)
        query = parse_qs(parts.query, keep_blank_values=False)
        host = (parts.netloc or "").lower()
        path = parts.path or ""

        for key in ("url", "u", "redirect", "redirect_url", "dest", "destination", "target"):
            value = query.get(key, [""])[0]
            if value.startswith(("http://", "https://")):
                return sanitize_job_url(unquote(value))

        if "linkedin.com" in host and "/redir/" in path:
            target = query.get("url", [""])[0]
            if target:
                return sanitize_job_url(unquote(target))

        clean_query = urlencode(
            [(k, v) for k, values in query.items() if not k.lower().startswith(("utm_", "trk", "ref", "fbclid", "gclid")) for v in values],
            doseq=True,
        )
        return urlunsplit((parts.scheme, parts.netloc, path.rstrip('/'), clean_query, ""))
    except Exception:
        return raw


def infer_source_from_url(url: str) -> str:
    host = (urlsplit(str(url or "")).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


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
    L3 Job discovery. Called by main.py:
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
        queries = self._build_queries(intent_plan)
        regions = self._resolve_locations(intent_plan)
        remote = intent_plan.get("geo_preferences", {}).get("remote", True)

        log.info("LeadScout starting: %d queries across %d regions", len(queries), len(regions))
        for i, q in enumerate(queries):
            log.info("  Query[%d]: %s", i, q)

        tasks = []
        for region in regions:
            providers = self._source_rotation_for_region(region)
            for query in queries:
                for provider in providers:
                    tasks.append(self._search_serper_organic(query, region, remote, provider))
                    tasks.append(self._search_tavily(query, region, remote, provider))

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
            lead.url = sanitize_job_url(lead.url)
            if not lead.source:
                lead.source = infer_source_from_url(lead.url)
            key = lead.url.strip().rstrip("/")
            if key and key not in seen:
                seen.add(key)
                unique.append(lead)

        log.info("LeadScout found %d unique leads (%d raw)", len(unique), len(leads))
        return [l.to_dict() for l in unique[: self.max_per_source * 4]]

    # Aliases
    find_jobs   = search_jobs
    scrape_jobs = search_jobs

    # ── Query builder ────────────────────────────────────────────────────────

    def _build_queries(self, intent_plan: dict) -> list[str]:
        roles    = intent_plan.get("target_roles", [])
        keywords = intent_plan.get("keywords", [])
        profile  = intent_plan.get("extracted_profile", {})

        # Gather all skills
        profile_skills: list[str] = []
        if isinstance(profile.get("skills"), list):
            profile_skills = [str(s) for s in profile["skills"]]

        all_keywords = list(dict.fromkeys(keywords + profile_skills))

        # Domain bucketing
        ai_ml = [k for k in all_keywords if any(t in k.lower() for t in [
            "ai", "ml", "llm", "gpt", "bert", "transformer", "pytorch", "tensorflow",
            "langchain", "langgraph", "genai", "generative", "diffusion", "rag",
            "vector", "embedding", "hugging", "openai", "fine-tun", "nlp",
            "computer vision", "deep learning", "neural", "mlops", "reinforcement",
        ])]
        cloud = [k for k in all_keywords if any(t in k.lower() for t in [
            "aws", "azure", "gcp", "sagemaker", "bedrock", "vertex", "cloud",
            "lambda", "kubernetes", "docker",
        ])]

        queries = []

        if roles:
            ai_str = " ".join(ai_ml[:4]) if ai_ml else " ".join(all_keywords[:4])
            queries.append(f"{roles[0]} {ai_str}")

        seniority = self._detect_seniority(profile, roles)
        if seniority and roles:
            queries.append(f"{seniority} {roles[0]}")

        for alt in self._alt_roles(roles)[:2]:
            ai_str = " ".join(ai_ml[:3]) if ai_ml else ""
            queries.append(f"{alt} {ai_str}".strip())

        if cloud and roles:
            queries.append(f"{roles[0]} {' '.join(cloud[:3])}")

        if any("gen" in k.lower() or "llm" in k.lower() for k in all_keywords):
            base = roles[0] if roles else "AI Engineer"
            queries.append(f"Generative AI {base} LLM")

        queries.append("Machine Learning Engineer GenAI LLM remote")

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
            "machine learning engineer": ["ML Engineer", "MLOps Engineer", "AI Engineer"],
            "data scientist":            ["Senior Data Scientist", "ML Researcher", "Applied Scientist"],
            "solution architect":        ["Solutions Architect AI", "Cloud AI Architect"],
            "genai":                     ["Generative AI Engineer", "LLM Engineer"],
            "ai architect":              ["AI Solutions Architect", "ML Architect"],
        }
        alts = []
        for role in roles:
            key = role.lower()
            for pattern, expansions in mapping.items():
                if pattern in key:
                    alts.extend(expansions)
        return list(dict.fromkeys(alts))

    def _resolve_locations(self, intent_plan: dict) -> list[dict[str, str]]:
        geo = intent_plan.get("geo_preferences", {}) or {}
        selector = str(geo.get("country_selector") or geo.get("region") or "").strip().upper()
        locs = [str(loc).strip() for loc in (geo.get("locations") or []) if str(loc).strip()]
        if selector and selector in COUNTRY_SEARCH_PRESETS and selector not in {loc.upper() for loc in locs}:
            locs.insert(0, selector)
        if not locs:
            default_selector = selector if selector in COUNTRY_SEARCH_PRESETS else "US"
            return [COUNTRY_SEARCH_PRESETS[default_selector]]

        resolved = []
        for loc in locs:
            upper = loc.upper()
            if upper in COUNTRY_SEARCH_PRESETS:
                resolved.append(COUNTRY_SEARCH_PRESETS[upper])
                continue
            match = next((preset for preset in COUNTRY_SEARCH_PRESETS.values() if loc.lower() in preset["label"].lower() or preset["location"].lower() in loc.lower()), None)
            resolved.append(match or {"label": loc, "location": loc, "gl": "us", "hl": "en"})
        seen = set()
        deduped = []
        for region in resolved:
            key = region["location"]
            if key not in seen:
                deduped.append(region)
                seen.add(key)
        return deduped

    def _source_rotation_for_region(self, region: dict[str, str]) -> list[dict[str, list[str]]]:
        selector = str(region.get("code") or "").upper()
        if not selector:
            selector = next((code for code, preset in COUNTRY_SEARCH_PRESETS.items() if preset.get("location") == region.get("location")), "US")
        return COUNTRY_SOURCE_ROTATION.get(selector, COUNTRY_SOURCE_ROTATION["US"])

    # ── Source: Serper /search (organic) ────────────────────────────────────

    async def _search_serper_organic(self, query: str, region: dict[str, str], remote: bool, provider: dict[str, list[str]]) -> list[JobLead]:
        if not SERPER_KEY:
            log.debug("Serper skipped — SERPER_API_KEY not set")
            return []
        try:
            loc_str = "remote" if remote else region["location"]
            site_domains = provider.get("domains") or JOB_BOARD_DOMAINS[:4]
            site_str = " OR ".join(f"site:{d}" for d in site_domains)
            search_q = f"{query} {loc_str} ({site_str})"
            payload  = {"q": search_q, "gl": region.get("gl", "us"), "hl": region.get("hl", "en"), "num": self.max_per_source}
            headers  = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}

            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    json=payload, headers=headers,
                )
                if resp.status_code != 200:
                    log.warning("Serper organic HTTP %d for: %s", resp.status_code, query)
                    return []
                data = resp.json()

            leads = []
            for r in data.get("organic", []):
                url   = r.get("link", "")
                title = r.get("title", "")
                if not url or not title:
                    continue
                if any(skip in url for skip in SKIP_PATHS):
                    continue
                if not any(d in url for d in site_domains):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = r.get("displayLink", ""),
                    url         = sanitize_job_url(url),
                    description = r.get("snippet", "")[:500],
                    source      = provider.get("label", "serper_organic").lower(),
                    remote      = "remote" in (r.get("snippet", "") + url).lower(),
                ))
            log.info("Serper organic: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Serper organic error for '%s': %s", query, exc)
            return []

    # ── Source: Tavily ───────────────────────────────────────────────────────

    async def _search_tavily(self, query: str, region: dict[str, str], remote: bool, provider: dict[str, list[str]]) -> list[JobLead]:
        if not TAVILY_KEY:
            log.debug("Tavily skipped — TAVILY_API_KEY not set")
            return []
        try:
            loc_str = "remote" if remote else region["location"]
            site_domains = provider.get("domains") or JOB_BOARD_DOMAINS[:4]
            payload = {
                "api_key":         TAVILY_KEY,
                "query":           f"{query} {loc_str} job opening apply now {provider.get('label', '')}",
                "search_depth":    "basic",
                "max_results":     self.max_per_source,
                "include_domains": site_domains,
            }
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post("https://api.tavily.com/search", json=payload)
                if resp.status_code != 200:
                    log.warning("Tavily HTTP %d for: %s", resp.status_code, query)
                    return []
                data = resp.json()

            leads = []
            for r in data.get("results", []):
                url   = r.get("url", "")
                title = r.get("title", "")
                if not url or not title:
                    continue
                if any(skip in url for skip in SKIP_PATHS):
                    continue
                if not any(d in url for d in site_domains):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = "",
                    url         = sanitize_job_url(url),
                    description = r.get("content", "")[:500],
                    source      = provider.get("label", "tavily").lower(),
                    remote      = "remote" in (r.get("content", "") + url).lower(),
                ))
            log.info("Tavily: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Tavily error for '%s': %s", query, exc)
            return []
