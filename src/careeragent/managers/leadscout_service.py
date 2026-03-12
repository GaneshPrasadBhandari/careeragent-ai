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
from datetime import datetime, timedelta, timezone
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, urlparse
from typing import Optional

import httpx

log = logging.getLogger("leadscout")

REQUEST_TIMEOUT = 20.0

TOP_SOURCE_QUOTAS = {
    "linkedin.com": 2,
    "indeed.com": 2,
    "glassdoor.com": 1,
    "myvisajobs.com": 1,
    "ziprecruiter.com": 1,
    "greenhouse.io": 1,
    "lever.co": 1,
    "myworkdayjobs.com": 1,
}

JOB_BOARD_DOMAINS = [
    "linkedin.com/jobs",
    "indeed.com",
    "glassdoor.com",
    "myvisajobs.com",
    "ziprecruiter.com",
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

SKIP_PATHS = ["/blog/", "/news/", "/about", "/company", "/press", "/learn"]


def _normalize_result_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)

    if parsed.netloc.endswith("google.com") and parsed.path == "/url":
        target = parse_qs(parsed.query).get("q", [""])[0].strip()
        if target:
            parsed = urlparse(target)

    if parsed.scheme not in {"http", "https"}:
        return ""
    path = parsed.path.rstrip("/")
    base = f"{parsed.scheme}://{parsed.netloc}{path}"

    # Keep required job-identifying query params for job boards where the
    # canonical listing link depends on them (e.g., Indeed/Glassdoor).
    host = parsed.netloc.lower()
    query_map = parse_qs(parsed.query, keep_blank_values=False)
    keep_params: list[str] = []
    if "indeed.com" in host:
        keep_params = ["jk", "vjs"]
    elif "glassdoor.com" in host:
        keep_params = ["jl", "joblistingid"]

    if keep_params:
        compact = "&".join(
            f"{key}={query_map[key][0]}"
            for key in keep_params
            if query_map.get(key) and str(query_map[key][0]).strip()
        )
        if compact:
            return f"{base}?{compact}"
    return base


def _is_plausible_job_link(url: str) -> bool:
    low = str(url or "").lower()
    if not low:
        return False
    if any(token in low for token in ["/search", "/jobs/demo", "?q="]):
        return False

    parsed = urlparse(low)
    host, path, query = parsed.netloc, parsed.path, parsed.query
    if "linkedin.com" in host:
        return "/jobs/view" in path
    if "indeed.com" in host:
        return "/viewjob" in path
    if "glassdoor.com" in host:
        return "joblistingid=" in query or "-job" in path
    if any(d in host for d in ("greenhouse.io", "lever.co", "workday", "myworkdayjobs", "icims.com", "jobvite.com", "smartrecruiters.com", "ziprecruiter.com", "myvisajobs.com")):
        return "/job" in path
    return True


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
    posted_hours_ago: Optional[int] = None

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
        self.last_search_diagnostics: dict = {
            "providers": {},
            "counts": {},
            "fallback_reason": None,
        }

    # ── Entry point ─────────────────────────────────────────────────────────

    async def search_jobs(self, intent_plan: dict) -> list[dict]:
        serper_key = str(os.getenv("SERPER_API_KEY", "")).strip()
        tavily_key = str(os.getenv("TAVILY_API_KEY", "")).strip()
        queries  = self._build_queries(intent_plan)
        location = self._resolve_location(intent_plan)
        remote   = intent_plan.get("geo_preferences", {}).get("remote", True)
        diagnostics: dict = {
            "providers": {
                "serper": bool(serper_key),
                "tavily": bool(tavily_key),
                "remotive": True,
            },
            "counts": {"serper_organic": 0, "tavily": 0, "remotive": 0},
            "fallback_reason": None,
        }

        log.info("LeadScout starting: %d queries, location='%s'", len(queries), location)
        for i, q in enumerate(queries):
            log.info("  Query[%d]: %s", i, q)

        # Run all queries concurrently
        tasks = []
        for query in queries:
            tasks.append(self._search_serper_organic(query, location, remote, serper_key=serper_key))
            tasks.append(self._search_tavily(query, location, remote, tavily_key=tavily_key))
            tasks.append(self._search_remotive(query, location, remote))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        leads: list[JobLead] = []
        for batch in results:
            if isinstance(batch, Exception):
                log.warning("LeadScout source error: %s", batch)
                continue
            if isinstance(batch, list):
                for lead in batch:
                    diagnostics["counts"][str(lead.source or "unknown")] = diagnostics["counts"].get(str(lead.source or "unknown"), 0) + 1
                leads.extend(batch)

        # Deduplicate by canonical URL
        seen, unique = set(), []
        for lead in leads:
            key = lead.url.strip().rstrip("/")
            if key and key not in seen:
                seen.add(key)
                unique.append(lead)

        recency_hours = float(intent_plan.get("recency_hours") or 24.0)
        unique = self._annotate_posting_age(unique)
        unique = self._filter_by_role_relevance(unique, intent_plan=intent_plan)
        unique = self._filter_by_recency(unique, recency_hours=recency_hours)
        unique = self._enforce_source_quotas(unique, quota_targets=TOP_SOURCE_QUOTAS)
        if not unique:
            provider_flags = diagnostics["providers"]
            if not provider_flags.get("serper") and not provider_flags.get("tavily"):
                diagnostics["fallback_reason"] = "SERPER_API_KEY/TAVILY_API_KEY missing; using demo fallback when remotive has no matching jobs."
            else:
                diagnostics["fallback_reason"] = "Providers responded but returned no matching jobs after quality filters."
        self.last_search_diagnostics = diagnostics

        log.info("LeadScout found %d unique leads (%d raw)", len(unique), len(leads))
        return [l.to_dict() for l in unique[: self.max_per_source * 4]]

    @staticmethod
    def _parse_posted_datetime(value: str) -> Optional[datetime]:
        txt = str(value or "").strip().lower()
        if not txt:
            return None

        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(txt, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        iso = txt.replace("z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass

        rel = re.search(r"(\d+)\s*(hour|hr|day|week|month)s?\s*ago", txt)
        if rel:
            qty = int(rel.group(1))
            unit = rel.group(2)
            if unit in {"hour", "hr"}:
                return datetime.now(timezone.utc) - timedelta(hours=qty)
            if unit == "day":
                return datetime.now(timezone.utc) - timedelta(days=qty)
            if unit == "week":
                return datetime.now(timezone.utc) - timedelta(weeks=qty)
            if unit == "month":
                return datetime.now(timezone.utc) - timedelta(days=30 * qty)

        if "today" in txt or "just posted" in txt:
            return datetime.now(timezone.utc)
        if "yesterday" in txt:
            return datetime.now(timezone.utc) - timedelta(days=1)
        return None

    def _filter_by_recency(self, leads: list[JobLead], *, recency_hours: float) -> list[JobLead]:
        if recency_hours <= 0:
            return leads
        cutoff = datetime.now(timezone.utc) - timedelta(hours=recency_hours)
        filtered: list[JobLead] = []
        for lead in leads:
            if lead.posted_hours_ago is not None:
                parsed = datetime.now(timezone.utc) - timedelta(hours=max(0, int(lead.posted_hours_ago)))
            else:
                candidate = lead.posted_date or lead.description
                parsed = self._parse_posted_datetime(candidate)
            if parsed is None or parsed >= cutoff:
                filtered.append(lead)
        return filtered

    def _annotate_posting_age(self, leads: list[JobLead]) -> list[JobLead]:
        now = datetime.now(timezone.utc)
        for lead in leads:
            candidate = lead.posted_date or lead.description
            parsed = self._parse_posted_datetime(candidate)
            if parsed is None:
                continue
            age_hours = int(max(0, (now - parsed).total_seconds() // 3600))
            lead.posted_hours_ago = age_hours
        return leads

    def _filter_by_role_relevance(self, leads: list[JobLead], *, intent_plan: dict) -> list[JobLead]:
        raw_roles = [str(r).strip() for r in (intent_plan.get("target_roles") or []) if str(r).strip()]
        if not raw_roles:
            return leads

        role_aliases: dict[str, list[str]] = {
            "ai engineer": ["ai engineer", "artificial intelligence engineer", "applied ai engineer"],
            "ai solution architect": ["ai solution architect", "ai solutions architect", "solutions architect ai", "ai architect"],
            "genai sol architect": ["genai architect", "generative ai architect", "llm architect", "genai solution architect"],
            "principal data scientist": ["principal data scientist", "lead data scientist", "staff data scientist"],
        }

        phrase_bank: list[str] = []
        for role in raw_roles:
            low = role.lower()
            phrase_bank.extend(role_aliases.get(low, [low]))

        phrase_bank = list(dict.fromkeys(p for p in phrase_bank if p))
        generic_tokens = {"engineer", "scientist", "architect", "developer", "principal", "senior", "staff", "lead", "solution", "solutions", "data"}
        scored: list[tuple[float, JobLead]] = []
        for lead in leads:
            text = " ".join([lead.title, lead.description, lead.company]).lower()
            best = 0.0
            for phrase in phrase_bank:
                tokens = [t for t in re.split(r"\W+", phrase) if len(t) >= 3]
                if not tokens:
                    continue
                overlap_tokens = [tok for tok in tokens if tok in text]
                overlap = len(overlap_tokens)
                phrase_score = overlap / len(tokens)
                if phrase in text:
                    phrase_score = min(1.0, phrase_score + 0.35)
                elif overlap_tokens and not any(tok not in generic_tokens for tok in overlap_tokens):
                    phrase_score = min(phrase_score, 0.34)
                best = max(best, phrase_score)
            scored.append((best, lead))

        strong = [lead for score, lead in scored if score >= 0.5]
        if strong:
            return strong

        scored.sort(key=lambda item: item[0], reverse=True)
        floor = max(12, int(len(leads) * 0.25))
        return [lead for _, lead in scored[:floor]]

    @staticmethod
    def _source_domain(lead: JobLead) -> str:
        host = (urlparse(lead.url).netloc or "").lower()
        return host[4:] if host.startswith("www.") else host

    def _enforce_source_quotas(self, leads: list[JobLead], *, quota_targets: dict[str, int]) -> list[JobLead]:
        by_source: dict[str, list[JobLead]] = {}
        for lead in leads:
            by_source.setdefault(self._source_domain(lead), []).append(lead)

        selected: list[JobLead] = []
        selected_urls: set[str] = set()

        for source, target in quota_targets.items():
            inventory: list[JobLead] = []
            for key, items in by_source.items():
                if source in key:
                    inventory.extend(items)
            for lead in inventory[: max(0, int(target))]:
                if lead.url not in selected_urls:
                    selected.append(lead)
                    selected_urls.add(lead.url)

        cap = self.max_per_source * 4
        for lead in leads:
            if len(selected) >= cap:
                break
            if lead.url in selected_urls:
                continue
            selected.append(lead)
            selected_urls.add(lead.url)

        return selected

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
            for role in roles[:4]:
                queries.append(f"{role} {ai_str}".strip())

        seniority = self._detect_seniority(profile, roles)
        if seniority and roles:
            for role in roles[:3]:
                queries.append(f"{seniority} {role}".strip())

        for alt in self._alt_roles(roles)[:2]:
            ai_str = " ".join(ai_ml[:3]) if ai_ml else ""
            queries.append(f"{alt} {ai_str}".strip())

        if cloud and roles:
            for role in roles[:2]:
                queries.append(f"{role} {' '.join(cloud[:3])}".strip())

        if any("gen" in k.lower() or "llm" in k.lower() for k in all_keywords) and roles:
            for role in roles[:2]:
                queries.append(f"Generative AI {role} LLM")

        seen_q: set[str] = set()
        final: list[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen_q:
                seen_q.add(q)
                final.append(q)
                if len(final) >= 12:
                    break

        return final or ["AI Engineer Python remote", "AI Solution Architect remote"]

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

    def _resolve_location(self, intent_plan: dict) -> str:
        geo  = intent_plan.get("geo_preferences", {})
        locs = geo.get("locations", [])
        return locs[0] if locs else "United States"

    # ── Source: Serper /search (organic) ────────────────────────────────────

    async def _search_serper_organic(self, query: str, location: str, remote: bool, *, serper_key: str) -> list[JobLead]:
        if not serper_key:
            log.debug("Serper skipped — SERPER_API_KEY not set")
            return []
        try:
            loc_str  = "remote" if remote else location
            site_str = " OR ".join(f"site:{d}" for d in JOB_BOARD_DOMAINS)
            search_q = f"{query} {loc_str} ({site_str})"
            payload  = {"q": search_q, "gl": "us", "hl": "en", "num": self.max_per_source}
            headers  = {"X-API-KEY": serper_key, "Content-Type": "application/json"}

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
                url   = _normalize_result_url(r.get("link", ""))
                title = r.get("title", "")
                if not url or not title:
                    continue
                if any(skip in url for skip in SKIP_PATHS):
                    continue
                if not any(d in url for d in JOB_BOARD_DOMAINS):
                    continue
                if not _is_plausible_job_link(url):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = r.get("displayLink", ""),
                    url         = url,
                    description = r.get("snippet", "")[:500],
                    posted_date = r.get("date", ""),
                    source      = "serper_organic",
                    remote      = "remote" in (r.get("snippet", "") + url).lower(),
                ))
            log.info("Serper organic: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Serper organic error for '%s': %s", query, exc)
            return []

    # ── Source: Tavily ───────────────────────────────────────────────────────

    async def _search_tavily(self, query: str, location: str, remote: bool, *, tavily_key: str) -> list[JobLead]:
        if not tavily_key:
            log.debug("Tavily skipped — TAVILY_API_KEY not set")
            return []
        try:
            loc_str = "remote" if remote else location
            payload = {
                "api_key":         tavily_key,
                "query":           f"{query} {loc_str} job opening apply now",
                "search_depth":    "basic",
                "max_results":     self.max_per_source,
                "include_domains": JOB_BOARD_DOMAINS,
            }
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post("https://api.tavily.com/search", json=payload)
                if resp.status_code != 200:
                    log.warning("Tavily HTTP %d for: %s", resp.status_code, query)
                    return []
                data = resp.json()

            leads = []
            for r in data.get("results", []):
                url   = _normalize_result_url(r.get("url", ""))
                title = r.get("title", "")
                if not url or not title:
                    continue
                if any(skip in url for skip in SKIP_PATHS):
                    continue
                if not any(d in url for d in JOB_BOARD_DOMAINS):
                    continue
                if not _is_plausible_job_link(url):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = "",
                    url         = url,
                    description = r.get("content", "")[:500],
                    posted_date = r.get("published_date", ""),
                    source      = "tavily",
                    remote      = "remote" in (r.get("content", "") + url).lower(),
                ))
            log.info("Tavily: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Tavily error for '%s': %s", query, exc)
            return []

    async def _search_remotive(self, query: str, location: str, remote: bool) -> list[JobLead]:
        try:
            search_term = " ".join(str(query or "").split()[:6]).strip() or "software engineer"
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.get("https://remotive.com/api/remote-jobs", params={"search": search_term})
                if resp.status_code != 200:
                    log.warning("Remotive HTTP %d for: %s", resp.status_code, query)
                    return []
                data = resp.json()

            leads: list[JobLead] = []
            for row in data.get("jobs", [])[: self.max_per_source]:
                url = _normalize_result_url(row.get("url", ""))
                title = str(row.get("title", "")).strip()
                if not url or not title:
                    continue
                lead_location = str(row.get("candidate_required_location") or "Remote")
                if not remote and location and location.lower() not in lead_location.lower():
                    continue
                posted = str(row.get("publication_date") or "")
                company = str(row.get("company_name") or "")
                leads.append(JobLead(
                    id=re.sub(r"\W+", "_", f"{company}_{title}")[:40],
                    title=title,
                    company=company,
                    url=url,
                    location=lead_location,
                    description=str(row.get("description", ""))[:500],
                    posted_date=posted,
                    source="remotive",
                    remote=True,
                ))
            log.info("Remotive: %d leads for: %s", len(leads), query)
            return leads
        except Exception as exc:
            log.error("Remotive error for '%s': %s", query, exc)
            return []
