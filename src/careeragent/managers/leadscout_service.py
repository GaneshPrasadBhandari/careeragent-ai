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
RANKING_TIMEOUT_SECONDS = 8.0

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
    # Normalize to https to avoid browser "connection is not private" for legacy http links.
    base = f"https://{parsed.netloc}{path}"

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


def _curated_query_url(domain_path: str, query: str) -> str:
    """Build resilient, openable board-search links for curated backfill rows."""
    domain = str(domain_path or "").strip().lower().strip("/")
    if not domain:
        return ""

    # Domains that commonly break behind `www.` (e.g., jobs.lever.co).
    if domain.startswith(("jobs.", "boards.")):
        host = domain
    elif domain == "myworkdayjobs.com":
        host = domain
    else:
        host = f"www.{domain}"

    if "linkedin" in domain:
        return f"https://{host}/?keywords={query}"
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
        return f"https://jobs.lever.co/?q={query}"
    if "myworkdayjobs.com" in domain:
        # myworkdayjobs root cannot serve cross-tenant searches; use a stable site-search.
        return f"https://www.google.com/search?q=site%3Amyworkdayjobs.com+{query}"
    return f"https://{host}?q={query}"


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
    if "greenhouse.io" in host:
        return "/jobs/" in path
    if "lever.co" in host:
        return "/jobs/" in path
    if any(d in host for d in ("workday", "myworkdayjobs", "icims.com", "jobvite.com", "smartrecruiters.com", "ziprecruiter.com", "myvisajobs.com")):
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
        self._settings = Settings()
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
                company=dom.split('/')[0],
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
