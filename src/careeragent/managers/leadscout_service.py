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
from itertools import islice
from dataclasses import asdict, dataclass
from collections import Counter
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlencode, urlsplit, urlunsplit

import httpx

from careeragent.core.settings import Settings
from careeragent.tools.llm_tools import GeminiClient

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

CORE_SOURCE_ROTATION = [
    {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
    {"label": "Glassdoor", "domains": ["glassdoor.com"]},
    {"label": "Indeed", "domains": ["indeed.com"]},
    {"label": "ZipRecruiter", "domains": ["ziprecruiter.com"]},
    {"label": "MyVisaJobs", "domains": ["myvisajobs.com"]},
    {"label": "Greenhouse", "domains": ["boards.greenhouse.io", "greenhouse.io"]},
    {"label": "Lever", "domains": ["jobs.lever.co", "lever.co"]},
    {"label": "Google Jobs", "domains": ["google.com", "googleusercontent.com"]},
]

COUNTRY_SOURCE_ROTATION = {
    "US": CORE_SOURCE_ROTATION,
    "IN": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
        {"label": "ZipRecruiter", "domains": ["ziprecruiter.com"]},
        {"label": "Google Jobs", "domains": ["google.com", "googleusercontent.com"]},
        {"label": "Naukri", "domains": ["naukri.com"]},
        {"label": "Wellfound", "domains": ["wellfound.com", "angel.co"]},
        {"label": "Monster", "domains": ["monsterindia.com", "foundit.in"]},
        {"label": "Greenhouse", "domains": ["boards.greenhouse.io", "greenhouse.io"]},
    ],
    "EU": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
        {"label": "ZipRecruiter", "domains": ["ziprecruiter.com"]},
        {"label": "Greenhouse", "domains": ["boards.greenhouse.io", "greenhouse.io"]},
        {"label": "Lever", "domains": ["jobs.lever.co", "lever.co"]},
        {"label": "Google Jobs", "domains": ["google.com", "googleusercontent.com"]},
        {"label": "Wellfound", "domains": ["wellfound.com", "angel.co"]},
    ],
    "AU": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
        {"label": "ZipRecruiter", "domains": ["ziprecruiter.com"]},
        {"label": "Greenhouse", "domains": ["boards.greenhouse.io", "greenhouse.io"]},
        {"label": "Lever", "domains": ["jobs.lever.co", "lever.co"]},
        {"label": "Google Jobs", "domains": ["google.com", "googleusercontent.com"]},
        {"label": "Wellfound", "domains": ["wellfound.com", "angel.co"]},
    ],
    "UAE": [
        {"label": "LinkedIn", "domains": ["linkedin.com/jobs"]},
        {"label": "Indeed", "domains": ["indeed.com"]},
        {"label": "Glassdoor", "domains": ["glassdoor.com"]},
        {"label": "ZipRecruiter", "domains": ["ziprecruiter.com"]},
        {"label": "Greenhouse", "domains": ["boards.greenhouse.io", "greenhouse.io"]},
        {"label": "Lever", "domains": ["jobs.lever.co", "lever.co"]},
        {"label": "Google Jobs", "domains": ["google.com", "googleusercontent.com"]},
        {"label": "Wellfound", "domains": ["wellfound.com", "angel.co"]},
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


def _normalize_job_identity(title: str, company: str, location: str = "") -> str:
    clean = []
    for value in (title, company, location):
        txt = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
        txt = re.sub(r"\b(remote|hybrid|onsite|on site|usa|united states|india|europe|australia|uae)\b", "", txt).strip()
        if txt:
            clean.append(re.sub(r"\s+", " ", txt))
    return " | ".join(clean[:3])


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
        self._settings = Settings()
        self._llm = GeminiClient(self._settings, model=os.getenv("CAREERAGENT_REASONING_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-1.5-flash")
        self.last_search_telemetry: dict[str, Any] = {}

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

        # Deduplicate by URL first, then collapse obvious same-job duplicates across portals.
        seen, unique = set(), []
        for lead in leads:
            lead.url = sanitize_job_url(lead.url)
            if not lead.source:
                lead.source = infer_source_from_url(lead.url)
            key = lead.url.strip().rstrip("/")
            if key and key not in seen:
                seen.add(key)
                unique.append(lead)

        quota_targets = self._build_source_quota_targets(regions)
        diversified = self._dedupe_similar_jobs(unique)
        diversified = self._enforce_source_quotas(diversified, quota_targets=quota_targets)
        self.last_search_telemetry = {
            "source_counts": dict(Counter((lead.source or infer_source_from_url(lead.url) or "unknown") for lead in diversified)),
            "source_quota_targets": quota_targets,
            "queries": queries,
            "regions": [region.get("location") for region in regions],
            "raw": len(leads),
            "unique": len(unique),
            "usable": len(diversified),
        }
        log.info("LeadScout found %d diversified leads (%d unique / %d raw)", len(diversified), len(unique), len(leads))
        return [l.to_dict() for l in diversified[: self.max_per_source * 4]]

    # Aliases
    find_jobs   = search_jobs
    scrape_jobs = search_jobs

    # ── Query builder ────────────────────────────────────────────────────────

    def _build_queries(self, intent_plan: dict) -> list[str]:
        roles = [str(r).strip() for r in (intent_plan.get("target_roles") or []) if str(r).strip()]
        keywords = [str(k).strip() for k in (intent_plan.get("keywords") or []) if str(k).strip()]
        profile = intent_plan.get("extracted_profile", {})
        self_learning_context = str(intent_plan.get("self_learning_context") or "").strip()

        profile_skills = [str(s).strip() for s in (profile.get("skills") or []) if str(s).strip()]
        all_keywords = list(dict.fromkeys(keywords + profile_skills))
        seed_role = roles[0] if roles else "AI Engineer"
        llm_variants = self._semantic_role_variants(
            seed_role=seed_role,
            profile=profile,
            keywords=all_keywords,
            self_learning_context=self_learning_context,
        )

        seniority = self._detect_seniority(profile, roles)
        ai_ml = [k for k in all_keywords if any(t in k.lower() for t in [
            "ai", "ml", "llm", "gpt", "bert", "transformer", "pytorch", "tensorflow",
            "langchain", "langgraph", "genai", "generative", "diffusion", "rag",
            "vector", "embedding", "hugging", "openai", "anthropic", "gemini", "fine-tun", "nlp",
        ])]
        cloud = [k for k in all_keywords if any(t in k.lower() for t in [
            "aws", "azure", "gcp", "sagemaker", "bedrock", "vertex", "cloud", "lambda", "kubernetes", "docker",
        ])]

        queries: list[str] = []
        queries.extend(llm_variants)
        if seniority:
            queries.append(f"{seniority} {seed_role}")
        for alt in self._alt_roles(roles)[:4]:
            queries.append(f"{alt} {' '.join(ai_ml[:3])}".strip())
        if cloud:
            queries.append(f"{seed_role} {' '.join(cloud[:3])}".strip())
        if ai_ml:
            queries.append(f"{seed_role} {' '.join(ai_ml[:4])}".strip())

        final: list[str] = []
        seen_q: set[str] = set()
        for q in queries:
            q = re.sub(r"\s+", " ", str(q or "")).strip()
            if q and q not in seen_q:
                seen_q.add(q)
                final.append(q)
            if len(final) >= 10:
                break
        return final or ["AI Engineer Python remote", "Machine Learning Engineer remote"]

    def _semantic_role_variants(self, *, seed_role: str, profile: dict, keywords: list[str], self_learning_context: str = "") -> list[str]:
        prompt = (
            "Generate 8 JSON array search variations for job discovery. "
            "Focus on semantic equivalents, adjacent titles, and architecture/leadership variations. "
            "Keep each variation under 12 words and usable as a search query. "
            "Treat Senior, Lead, Principal, and Architect as interchangeable for this candidate. "
            f"Seed role: {seed_role}. Candidate skills: {', '.join(keywords[:20]) or 'not provided'}. "
            f"Long-term reviewer learning context: {self_learning_context[:400] or 'none'}."
        )
        payload = self._llm.generate_json(prompt, temperature=0.2, max_tokens=300)
        variants: list[str] = []
        if isinstance(payload, dict):
            raw = payload.get("variations") or payload.get("queries") or payload.get("roles") or []
            if isinstance(raw, list):
                variants = [str(item).strip() for item in raw if str(item).strip()]
        if not variants:
            variants = [
                seed_role,
                f"Senior {seed_role}",
                f"Lead {seed_role}",
                f"Principal {seed_role}",
                f"Architect {seed_role}",
                f"{seed_role} platform",
                f"{seed_role} distributed systems",
                f"{seed_role} generative ai",
                f"Applied {seed_role}",
                f"{seed_role} machine learning",
            ]
        return list(islice(dict.fromkeys(v for v in variants if v), 10))

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

    def _build_source_quota_targets(self, regions: list[dict[str, str]]) -> dict[str, int]:
        quotas: dict[str, int] = {}
        for region in regions or [COUNTRY_SEARCH_PRESETS["US"]]:
            for provider in self._source_rotation_for_region(region)[:8]:
                quotas.setdefault(provider["label"].lower(), 1)
        return quotas

    def _dedupe_similar_jobs(self, leads: list[JobLead]) -> list[JobLead]:
        unique: list[JobLead] = []
        seen_identity: set[str] = set()
        for lead in leads:
            identity = _normalize_job_identity(lead.title, lead.company, lead.location)
            if identity and identity in seen_identity:
                continue
            if identity:
                seen_identity.add(identity)
            unique.append(lead)
        return unique

    def _enforce_source_quotas(self, leads: list[JobLead], *, quota_targets: dict[str, int]) -> list[JobLead]:
        by_source: dict[str, list[JobLead]] = {}
        for lead in leads:
            source = str(lead.source or infer_source_from_url(lead.url) or "unknown").lower()
            by_source.setdefault(source, []).append(lead)

        selected: list[JobLead] = []
        selected_urls: set[str] = set()

        for source, target in quota_targets.items():
            inventory = [lead for key, entries in by_source.items() if source in key for lead in entries]
            for lead in inventory[: max(1, int(target))]:
                if lead.url and lead.url not in selected_urls:
                    selected.append(lead)
                    selected_urls.add(lead.url)

        cap = self.max_per_source * 4
        for lead in leads:
            if len(selected) >= cap:
                break
            if not lead.url or lead.url in selected_urls:
                continue
            selected.append(lead)
            selected_urls.add(lead.url)
        return selected

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
                direct_url = sanitize_job_url(url)
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = r.get("displayLink", ""),
                    url         = direct_url,
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
                direct_url = sanitize_job_url(url)
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = "",
                    url         = direct_url,
                    description = r.get("content", "")[:500],
                    source      = provider.get("label", "tavily").lower(),
                    remote      = "remote" in (r.get("content", "") + url).lower(),
                ))
            log.info("Tavily: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Tavily error for '%s': %s", query, exc)
            return []
