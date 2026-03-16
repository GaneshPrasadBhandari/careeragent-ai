"""
LeadScout Service — managers/leadscout_service.py
===================================================
Uses httpx (already in deps) instead of aiohttp.
Serper /jobs removed — returns 404 on this plan.
Uses Serper /search organic + Tavily as primary sources.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime, timedelta, timezone
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, quote_plus, urlparse
from typing import Optional

import httpx

from careeragent.core.settings import Settings
from careeragent.tools.llm_tools import GeminiClient

log = logging.getLogger("leadscout")

REQUEST_TIMEOUT = 20.0
LLM_QUERY_TIMEOUT_SECONDS = 8.0
SEARCH_TASK_TIMEOUT_SECONDS = 25.0
RANKING_TIMEOUT_SECONDS = 300.0

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

SUPPORTED_MIRROR_SITES = [
    "linkedin.com/jobs/view",
    "indeed.com/viewjob",
    "glassdoor.com/job",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "myworkdayjobs.com",
]

ROTATING_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


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
    # Normalize to https to avoid browser "connection is not private" for legacy http links.
    base = f"https://{parsed.netloc}{path}"

    # Canonicalize URLs by stripping query/fragments (tracking params, campaign tags).
    # This ensures the same job reached via different query strings dedupes reliably.
    return base


def _normalize_job_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip().lower())


def _composite_lead_key(lead: "JobLead") -> tuple[str, str]:
    company = re.sub(r"\s+", " ", str(lead.company or "").strip().lower())
    return company, _normalize_job_title(lead.title)


def _clean_company_name(raw: str) -> str:
    text = re.sub(r"\s+", " ", str(raw or "").strip())
    if not text:
        return ""
    low = text.lower()
    board_noise = {
        "linkedin", "linkedin.com", "indeed", "indeed.com", "glassdoor", "glassdoor.com",
        "ziprecruiter", "ziprecruiter.com", "myworkdayjobs", "workday", "jobs", "careers",
    }
    if low in board_noise:
        return ""
    return text


def _infer_company_name(*, title: str, company_hint: str, url: str) -> str:
    cleaned_hint = _clean_company_name(company_hint)
    if cleaned_hint:
        return cleaned_hint

    title_text = re.sub(r"\s+", " ", str(title or "").strip())
    patterns = [
        r"\bat\s+([A-Z][A-Za-z0-9&.,'\- ]{1,50})$",
        r"\|\s*([A-Z][A-Za-z0-9&.,'\- ]{1,50})$",
        r"-\s*([A-Z][A-Za-z0-9&.,'\- ]{1,50})$",
    ]
    for pat in patterns:
        m = re.search(pat, title_text)
        if m:
            inferred = _clean_company_name(m.group(1).strip(" -|"))
            if inferred:
                return inferred

    parsed = urlparse(str(url or "").strip())
    host = (parsed.netloc or "").lower().removeprefix("www.")
    if any(x in host for x in ("greenhouse.io", "lever.co", "myworkdayjobs.com", "workday.com")):
        first = [p for p in parsed.path.split("/") if p][:1]
        if first and first[0].lower() not in {"jobs", "job", "en-us", "recruiting"}:
            tenant = re.sub(r"[-_]+", " ", first[0]).strip()
            if tenant:
                return " ".join(tok.capitalize() for tok in tenant.split())
    return "Unknown company"


def _curated_query_url(domain_path: str, query: str) -> str:
    """Build resilient, openable board-search links for curated backfill rows."""
    raw_domain = str(domain_path or "").strip().lower()
    if not raw_domain:
        return ""
    if "://" not in raw_domain:
        parsed = urlparse(f"https://{raw_domain}")
    else:
        parsed = urlparse(raw_domain)
    host_only = (parsed.netloc or parsed.path.split("/")[0]).strip().lower()
    host_only = host_only.removeprefix("www.")
    path_hint = parsed.path or ""
    domain = host_only

    # Domains that commonly break behind `www.` (e.g., jobs.lever.co).
    if domain.startswith(("jobs.", "boards.")):
        host = domain
    elif domain == "myworkdayjobs.com":
        host = domain
    else:
        host = f"www.{domain}"

    if "linkedin" in domain:
        # Keep users anchored to LinkedIn Jobs search to avoid homepage/feed redirects.
        return f"https://{host}/jobs/search/?keywords={query}"
    if "indeed" in domain:
        return f"https://{host}/jobs?q={query}"
    if "glassdoor" in domain:
        # /Job/jobs.htm frequently 404s for direct query params; /Job/index.htm is stable.
        return f"https://{host}/Job/index.htm?sc.keyword={query}"
    if "ziprecruiter" in domain:
        return f"https://{host}/jobs-search?search={query}"
    if "greenhouse" in domain:
        return f"https://www.google.com/search?q=site%3Aboards.greenhouse.io+{query}"
    if "jobs.lever.co" in domain or "lever.co" in domain:
        # Root lever.co often lands on vendor homepage; use site search to jump to tenant pages.
        return f"https://www.google.com/search?q=site%3Ajobs.lever.co+{query}"
    if "myworkdayjobs.com" in domain:
        # myworkdayjobs root cannot serve cross-tenant searches; use a stable site-search.
        return f"https://www.google.com/search?q=site%3Amyworkdayjobs.com+{query}"
    if path_hint and path_hint != "/":
        return f"https://{host}{path_hint}?q={query}"
    return f"https://{host}?q={query}"


def _is_valid_job_url(url: str) -> bool:
    low = str(url or "").lower()
    if not low:
        return False
    blacklist = ("/jobs?", "/job/index.htm?sc.keyword", "search?q=", "results?")
    if any(token in low for token in blacklist):
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
    if "greenhouse.io" in host:
        return "/jobs/" in path or "/embed/job_app" in path
    if "lever.co" in host:
        return "/jobs/" in path
    if any(d in host for d in ("workday", "myworkdayjobs", "icims.com", "jobvite.com", "smartrecruiters.com", "ziprecruiter.com", "myvisajobs.com")):
        return "/job" in path
    return any(
        token in host
        for token in ("greenhouse.io", "lever.co", "myworkdayjobs.com", "workday.com", "icims.com", "jobvite.com", "smartrecruiters.com")
    ) and any(token in path for token in ("/job", "/jobs/", "/recruiting/"))


def _is_supported_mirror_board_url(url: str) -> bool:
    low = str(url or "").lower()
    return any(site in low for site in SUPPORTED_MIRROR_SITES)


def _looks_like_blocked_portal_response(status_code: int, body: str) -> bool:
    low = str(body or "").lower()
    return status_code in {401, 403, 404, 429, 503} or any(
        token in low
        for token in (
            "access denied",
            "bot detection",
            "security challenge",
            "verify you are human",
            "request blocked",
            "cloudflare",
        )
    )


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
        self._settings = Settings()
        self._ua_pool = list(ROTATING_USER_AGENTS)
        self._ua_cursor = random.randint(0, max(0, len(self._ua_pool) - 1)) if self._ua_pool else 0
        self._playwright_stealth = {
            "headless": True,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
            "context": {
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "color_scheme": "light",
            },
        }
        self._llm_clients: list[GeminiClient] = [
            GeminiClient(self._settings, model="gemini-1.5-flash"),
            GeminiClient(self._settings, model="gemini-1.5-pro"),
            GeminiClient(self._settings, model="gemini-2.0-flash-exp"),
        ]
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
        try:
            llm_queries = await asyncio.wait_for(
                asyncio.to_thread(self._llm_expand_queries, intent_plan, queries),
                timeout=LLM_QUERY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            llm_queries = []
            log.warning("LeadScout query expansion timed out after %.1fs; using baseline queries.", LLM_QUERY_TIMEOUT_SECONDS)
        if llm_queries:
            for q in llm_queries:
                if q not in queries:
                    queries.append(q)
            queries = queries[:18]
        location = self._resolve_location(intent_plan)
        remote   = intent_plan.get("geo_preferences", {}).get("remote", True)
        diagnostics: dict = {
            "providers": {
                "serper": bool(serper_key),
                "tavily": bool(tavily_key),
                "remotive": True,
                "llm_query_planner": bool(self._settings.GEMINI_API_KEY),
                "llm_ranker": bool(self._settings.GEMINI_API_KEY),
                "headless_stealth_ready": True,
            },
            "counts": {"serper_organic": 0, "tavily": 0, "remotive": 0},
            "fallback_reason": None,
        }

        log.info("LeadScout starting: %d queries, location='%s'", len(queries), location)
        for i, q in enumerate(queries):
            log.info("  Query[%d]: %s", i, q)

        # Run all queries concurrently
        tasks: list[asyncio.Task[list[JobLead]]] = []
        for query in queries:
            tasks.append(asyncio.create_task(self._search_serper_organic(query, location, remote, serper_key=serper_key)))
            tasks.append(asyncio.create_task(self._search_tavily(query, location, remote, tavily_key=tavily_key)))
            tasks.append(asyncio.create_task(self._search_remotive(query, location, remote)))

        done, pending = await asyncio.wait(tasks, timeout=SEARCH_TASK_TIMEOUT_SECONDS)
        if pending:
            log.warning(
                "LeadScout timed out waiting for %d source tasks after %.1fs; continuing with partial results.",
                len(pending),
                SEARCH_TASK_TIMEOUT_SECONDS,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        results: list[list[JobLead]] = []
        for task in done:
            if task.cancelled():
                continue
            try:
                batch = task.result()
            except Exception as exc:
                log.warning("LeadScout source error: %s", exc)
                continue
            results.append(batch)

        leads: list[JobLead] = []
        for batch in results:
            if isinstance(batch, list):
                for lead in batch:
                    diagnostics["counts"][str(lead.source or "unknown")] = diagnostics["counts"].get(str(lead.source or "unknown"), 0) + 1
                leads.extend(batch)

        if leads:
            leads = await self._validate_and_retry_links(leads, serper_key=serper_key)

        # Deduplicate by canonical URL + per-session composite identity
        # (company + normalized title) to drop repeated board mirrors.
        seen_urls, seen_composite, unique = set(), set(), []
        for lead in leads:
            normalized_url = _normalize_result_url(lead.url)
            lead.url = normalized_url or lead.url
            if normalized_url and normalized_url in seen_urls:
                continue
            composite_key = _composite_lead_key(lead)
            if composite_key in seen_composite:
                continue
            if normalized_url:
                seen_urls.add(normalized_url)
            seen_composite.add(composite_key)
            unique.append(lead)

        recency_hours = float(intent_plan.get("recency_hours") or 24.0)
        unique = self._annotate_posting_age(unique)
        unique = self._filter_by_role_relevance(unique, intent_plan=intent_plan)
        unique = self._filter_by_recency(unique, recency_hours=recency_hours)
        unique = self._enforce_source_quotas(unique, quota_targets=TOP_SOURCE_QUOTAS)

        target_count = int(intent_plan.get("max_jobs") or 80)
        if len(unique) < target_count:
            # If live providers produced little/no data, avoid flooding ranking with many
            # repetitive query URLs that are not direct application pages.
            backfill_target = target_count if unique else min(target_count, 30)
            unique = self._backfill_curated_search_urls(unique, intent_plan=intent_plan, target_count=backfill_target)
            if backfill_target < target_count:
                diagnostics["fallback_reason"] = (
                    f"Live providers returned low volume; limited non-direct query backfill to {backfill_target} rows "
                    "to keep rankings actionable."
                )

        try:
            unique = await asyncio.wait_for(
                asyncio.to_thread(self._rank_leads_hybrid, unique, intent_plan),
                timeout=RANKING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.warning("LeadScout ranking timed out after %.1fs; using unrated ordering.", RANKING_TIMEOUT_SECONDS)
        if not unique:
            provider_flags = diagnostics["providers"]
            if not provider_flags.get("serper") and not provider_flags.get("tavily"):
                diagnostics["fallback_reason"] = "SERPER_API_KEY/TAVILY_API_KEY missing; using demo fallback when remotive has no matching jobs."
            else:
                diagnostics["fallback_reason"] = "Providers responded but returned no matching jobs after quality filters."
        self.last_search_diagnostics = diagnostics

        log.info("LeadScout found %d unique leads (%d raw)", len(unique), len(leads))
        return [l.to_dict() for l in unique[: self.max_per_source * 4]]

    def _next_user_agent(self) -> str:
        if not self._ua_pool:
            return ""
        self._ua_cursor = (self._ua_cursor + 1) % len(self._ua_pool)
        return self._ua_pool[self._ua_cursor]

    async def _linkedin_fresh_context_resolve(self, url: str) -> tuple[bool, str]:
        """LinkedIn-only recovery: clear cookies in browser context, then retry URL."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return False, url

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
                context = await browser.new_context(locale="en-US")
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                final_url = str(page.url)
                if "linkedin.com/feed" in final_url.lower() or "/feed/" in final_url.lower():
                    await context.clear_cookies()
                    await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                    final_url = str(page.url)
                await context.close()
                await browser.close()
                return _is_valid_job_url(final_url), final_url
        except Exception:
            return False, url


    async def _resolve_redirect_or_block(self, url: str) -> tuple[bool, str, str]:
        """Return (ok, final_url, reason)."""
        headers = {
            "User-Agent": self._next_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
            final_url = str(response.url)
            body = (response.text or "")[:1200]
            blocked = _looks_like_blocked_portal_response(response.status_code, body)
            if blocked:
                return False, final_url or url, f"blocked_or_error_http_{response.status_code}"
            if "linkedin.com" in final_url.lower() and "/feed" in final_url.lower():
                recovered_ok, recovered_url = await self._linkedin_fresh_context_resolve(url)
                if recovered_ok and _is_valid_job_url(recovered_url):
                    return True, recovered_url, "ok_linkedin_cookie_reset"
                # Keep uncertain LinkedIn redirect results so L2 parser can classify later.
                return True, recovered_url or final_url or url, "linkedin_feed_redirect_kept_for_l2"
            return True, final_url or url, "ok"
        except Exception as exc:
            return False, url, f"resolve_error:{type(exc).__name__}"

    async def _search_retry_mirror_link(self, *, company: str, title: str, serper_key: str) -> str:
        if not serper_key:
            return ""
        company_q = str(company or "").strip()
        title_q = str(title or "").strip()
        query = (
            f"{company_q} {title_q} "
            "(site:linkedin.com/jobs/view OR site:indeed.com/viewjob OR site:boards.greenhouse.io OR site:jobs.lever.co OR site:myworkdayjobs.com)"
        ).strip()
        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json", "User-Agent": self._next_user_agent()}
        payload = {"q": query, "gl": "us", "hl": "en", "num": 8}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post("https://google.serper.dev/search", json=payload, headers=headers)
            if resp.status_code != 200:
                return ""
            data = resp.json()
            for row in data.get("organic", []):
                url = _normalize_result_url(row.get("link", ""))
                if url and _is_supported_mirror_board_url(url) and _is_valid_job_url(url):
                    return url
            return ""
        except Exception:
            return ""

    async def _validate_and_retry_links(self, leads: list[JobLead], *, serper_key: str) -> list[JobLead]:
        repaired: list[JobLead] = []
        for lead in leads:
            url = str(lead.url or "")
            if not url:
                repaired.append(lead)
                continue

            host = urlparse(url).netloc.lower()
            if not any(token in host for token in ("workday", "myworkdayjobs", "greenhouse", "lever", "icims", "smartrecruiters", "jobvite", "ashbyhq", "rippling")):
                repaired.append(lead)
                continue

            ok, final_url, reason = await self._resolve_redirect_or_block(url)
            if ok and _is_valid_job_url(final_url):
                lead.url = final_url
                repaired.append(lead)
                continue

            mirror = await self._search_retry_mirror_link(company=lead.company, title=lead.title, serper_key=serper_key)
            if mirror:
                lead.url = mirror
                lead.source = f"{lead.source}_search_retry" if lead.source else "search_retry"
                lead.description = (lead.description or "")[:420] + f" | search_retry_reason: {reason}"
            repaired.append(lead)
        return repaired

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

    def _llm_expand_queries(self, intent_plan: dict, base_queries: list[str]) -> list[str]:
        if not self._settings.GEMINI_API_KEY:
            return []
        roles = [str(r).strip() for r in (intent_plan.get("target_roles") or []) if str(r).strip()]
        skills = [str(k).strip() for k in (intent_plan.get("keywords") or []) if str(k).strip()][:20]
        profile = intent_plan.get("extracted_profile") or {}
        summary = str(profile.get("summary") or "")[:700]
        prompt = (
            "Generate up to 10 job-search queries for high-precision discovery. Return JSON only: "
            "{\"queries\":[...]}\n"
            f"Target roles: {roles}\n"
            f"Skills: {skills}\n"
            f"Profile summary: {summary}\n"
            f"Existing queries: {base_queries[:8]}\n"
            "Need a mix of exact-title, adjacent-title, and keyword-context queries for US/remote hiring."
        )
        for client in self._llm_clients:
            out = client.generate_json(prompt, temperature=0.2, max_tokens=600)
            if not isinstance(out, dict):
                continue
            queries = [str(q).strip() for q in (out.get("queries") or []) if str(q).strip()]
            if queries:
                return queries[:10]
        return []

    def _hybrid_relevance_score(self, lead: JobLead, intent_plan: dict) -> tuple[float, str]:
        roles = [str(r).lower().strip() for r in (intent_plan.get("target_roles") or []) if str(r).strip()]
        keywords = [str(k).lower().strip() for k in (intent_plan.get("keywords") or []) if str(k).strip()]
        text = " ".join([lead.title, lead.company, lead.description]).lower()

        role_hits = sum(1 for r in roles if r and r in text)
        kw_hits = sum(1 for k in keywords[:30] if k and k in text)
        semantic_proxy = 0.0
        if roles:
            semantic_proxy += min(1.0, role_hits / max(1, len(roles)))
        if keywords:
            semantic_proxy += min(1.0, kw_hits / max(3, min(len(keywords), 12)))
        semantic_proxy = semantic_proxy / 2.0

        context_bonus = 0.12 if lead.remote else 0.0
        if lead.posted_hours_ago is not None and int(lead.posted_hours_ago) <= 72:
            context_bonus += 0.12
        score = max(0.0, min(1.0, 0.75 * semantic_proxy + context_bonus))
        reason = f"role_hits={role_hits}, keyword_hits={kw_hits}, remote={lead.remote}, recency_h={lead.posted_hours_ago}"
        return score, reason

    def _llm_rank_boosts(self, leads: list[JobLead], intent_plan: dict) -> dict[str, float]:
        if not self._settings.GEMINI_API_KEY or not leads:
            return {}
        roles = [str(r).strip() for r in (intent_plan.get("target_roles") or []) if str(r).strip()]
        skills = [str(k).strip() for k in (intent_plan.get("keywords") or []) if str(k).strip()][:20]
        payload = [
            {"id": lead.id, "title": lead.title, "company": lead.company, "desc": lead.description[:220]}
            for lead in leads[:40]
        ]
        prompt = (
            "Rank these jobs for candidate fit from 0..1 using role alignment, skill overlap, and responsibility context. "
            "Return JSON only as {\"scores\":[{\"id\":...,\"score\":0.0,\"reason\":...}]}.\n"
            f"Target roles: {roles}\nSkills: {skills}\nJobs: {json.dumps(payload)}"
        )
        for client in self._llm_clients:
            out = client.generate_json(prompt, temperature=0.1, max_tokens=1200)
            if not isinstance(out, dict):
                continue
            scores = {}
            for row in (out.get("scores") or []):
                jid = str((row or {}).get("id") or "").strip()
                if not jid:
                    continue
                try:
                    val = float((row or {}).get("score") or 0.0)
                except Exception:
                    val = 0.0
                scores[jid] = max(0.0, min(1.0, val))
            if scores:
                return scores
        return {}

    def _rank_leads_hybrid(self, leads: list[JobLead], intent_plan: dict) -> list[JobLead]:
        if not leads:
            return leads
        llm_boost = self._llm_rank_boosts(leads, intent_plan)
        scored: list[tuple[float, JobLead]] = []
        for lead in leads:
            base, reason = self._hybrid_relevance_score(lead, intent_plan)
            boost = float(llm_boost.get(lead.id, 0.0))
            final = (0.72 * base) + (0.28 * boost if boost > 0 else 0.0)
            lead.description = (lead.description or "")[:430] + (" | rank_reason: " + reason)
            scored.append((final, lead))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [lead for _, lead in scored]

    def _backfill_curated_search_urls(self, leads: list[JobLead], *, intent_plan: dict, target_count: int) -> list[JobLead]:
        existing = list(leads)
        seen = {x.url for x in existing if x.url}
        roles = [str(r).strip() for r in (intent_plan.get("target_roles") or []) if str(r).strip()] or ["Software Engineer"]
        keywords = [str(k).strip() for k in (intent_plan.get("keywords") or []) if str(k).strip()][:8]
        domains = [
            "linkedin.com/jobs/search",
            "indeed.com/jobs",
            "glassdoor.com/Job/jobs.htm",
            "ziprecruiter.com/Jobs",
            "boards.greenhouse.io",
            "jobs.lever.co",
            "myworkdayjobs.com",
        ]
        idx = 0
        while len(existing) < target_count:
            role = roles[idx % len(roles)]
            kw = keywords[idx % len(keywords)] if keywords else ""
            dom = domains[idx % len(domains)]
            q = quote_plus((f"{role} {kw}").strip())
            url = _curated_query_url(dom, q)
            idx += 1
            if url in seen:
                continue
            seen.add(url)
            existing.append(JobLead(
                id=f"backfill_{idx:03d}",
                title=role,
                company="Unknown company",
                url=url,
                location="Remote" if intent_plan.get("geo_preferences", {}).get("remote", True) else self._resolve_location(intent_plan),
                remote=bool(intent_plan.get("geo_preferences", {}).get("remote", True)),
                description=f"Backfilled discovery query for {role}. Skills context: {', '.join(keywords[:4])}",
                source="query_backfill",
                posted_hours_ago=(idx % 72) + 1,
            ))
            if idx > target_count * 3:
                break
        return existing

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
        core_skills = [str(k).strip() for k in all_keywords if str(k).strip()][:21]

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

        for adj in self._adjacent_queries_from_skills(core_skills, limit=5):
            queries.append(adj)

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

    def _adjacent_queries_from_skills(self, skills: list[str], *, limit: int = 5) -> list[str]:
        bucket_map = {
            "mlops": ["MLOps Engineer", "Machine Learning Platform Engineer"],
            "kubernetes": ["Platform Engineer", "Cloud Platform Engineer"],
            "docker": ["DevOps Engineer", "Site Reliability Engineer"],
            "aws": ["Cloud Engineer", "Solutions Architect"],
            "azure": ["Azure AI Engineer", "Cloud Solutions Architect"],
            "gcp": ["Cloud Engineer", "Data Platform Engineer"],
            "langchain": ["LLM Engineer", "AI Application Engineer"],
            "langgraph": ["Agentic AI Engineer", "AI Platform Engineer"],
            "rag": ["Retrieval Engineer", "Applied AI Engineer"],
            "pytorch": ["ML Engineer", "Applied Scientist"],
            "tensorflow": ["ML Engineer", "Applied Scientist"],
            "airflow": ["Data Engineer", "ML Platform Engineer"],
            "spark": ["Data Engineer", "Data Platform Engineer"],
            "kafka": ["Streaming Data Engineer", "Backend Engineer"],
            "terraform": ["Cloud Infrastructure Engineer", "DevOps Engineer"],
            "snowflake": ["Analytics Engineer", "Data Engineer"],
            "databricks": ["Data Engineer", "ML Engineer"],
            "python": ["Backend Engineer", "Automation Engineer"],
            "sql": ["Data Engineer", "Analytics Engineer"],
            "nlp": ["NLP Engineer", "Applied Scientist"],
            "llm": ["LLM Engineer", "Generative AI Engineer"],
            "genai": ["Generative AI Engineer", "Applied AI Engineer"],
        }
        adjacent: list[str] = []
        seen: set[str] = set()
        for raw in skills[:21]:
            s = str(raw or "").strip().lower()
            if not s:
                continue
            for key, expansions in bucket_map.items():
                if key in s:
                    for role in expansions:
                        q = f"{role} remote United States"
                        if q.lower() not in seen:
                            seen.add(q.lower())
                            adjacent.append(q)
                            if len(adjacent) >= limit:
                                return adjacent
        return adjacent[:limit]

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
                if not _is_valid_job_url(url):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = _infer_company_name(title=title, company_hint=r.get("displayLink", ""), url=url),
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
                if not _is_valid_job_url(url):
                    continue
                leads.append(JobLead(
                    id          = re.sub(r"\W+", "_", title)[:40],
                    title       = title,
                    company     = _infer_company_name(title=title, company_hint="", url=url),
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
