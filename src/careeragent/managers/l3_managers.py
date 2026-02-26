from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState
from careeragent.tools.web_tools import JinaReader, RobotsGuard, SerperClient, canonical_url, extract_explicit_location, is_outside_target_geo


class LeadScout:
    """Description: LeadScout generates and executes search personas.
    Layer: L3
    Input: AgentState.search_personas
    Output: jobs_raw list
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.serper = SerperClient(settings)

    @staticmethod
    def _resolve_location(state: AgentState) -> str:
        pref_loc = (state.preferences.location or "").strip()
        profile_loc = str((state.extracted_profile or {}).get("contact", {}).get("location") or "").strip()
        return pref_loc or profile_loc or "United States, Remote"

    @staticmethod
    def _resolve_country_code(state: AgentState) -> str:
        raw = (state.preferences.country or "").strip().upper()
        if not raw:
            return "US"
        if raw in {"US", "USA", "UNITED STATES"}:
            return "US"
        if len(raw) == 2:
            return raw
        return "US"

    @staticmethod
    def _profile_skill_terms(state: AgentState, limit: int = 36) -> List[str]:
        skills = (state.extracted_profile or {}).get("skills") or []
        cleaned: List[str] = []
        seen = set()
        for skill in skills:
            sk = str(skill or "").strip()
            if len(sk) < 2:
                continue
            key = sk.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(sk)
            if len(cleaned) >= limit:
                break
        return cleaned

    @staticmethod
    def _cluster_skills(skills: List[str]) -> Dict[str, List[str]]:
        taxonomy = {
            "cloud": ["aws", "azure", "gcp", "cloud", "kubernetes", "docker", "ec2"],
            "data": ["sql", "spark", "airflow", "dbt", "etl", "warehouse", "snowflake"],
            "backend": ["python", "java", "golang", "node", "api", "microservices", "fastapi"],
            "frontend": ["react", "typescript", "javascript", "next", "ui", "css", "html"],
            "ai_ml": ["llm", "langchain", "langgraph", "ml", "ai", "pytorch", "tensorflow", "rag"],
            "devops": ["terraform", "ansible", "jenkins", "github actions", "ci/cd", "prometheus"],
        }
        clusters: Dict[str, List[str]] = {k: [] for k in taxonomy}
        clusters["general"] = []
        for sk in skills:
            low = sk.lower()
            matched = False
            for group, hints in taxonomy.items():
                if any(h in low for h in hints):
                    clusters[group].append(sk)
                    matched = True
            if not matched:
                clusters["general"].append(sk)
        return {k: v for k, v in clusters.items() if v}

    def build_query(self, state: AgentState) -> str:
        persona = next((p for p in state.search_personas if p.persona_id == state.active_persona_id), None)
        if not persona:
            persona = state.search_personas[0]
            state.active_persona_id = persona.persona_id

        must_terms = [x for x in persona.must_include if x]
        neg = " ".join([f"-{x}" for x in persona.negative_terms if x])

        broadening_level = int(state.query_modifiers.get("broadening_level") or 0)
        if broadening_level > 0 and must_terms:
            keep_count = max(1, len(must_terms) - broadening_level)
            must_terms = must_terms[:keep_count]

        must = " ".join([f'"{x}"' for x in must_terms])

        profile_skills = self._profile_skill_terms(state, limit=36)
        clusters = self._cluster_skills(profile_skills)
        cluster_exprs: List[str] = []
        for _, items in clusters.items():
            cluster_exprs.append("(" + " OR ".join([f'"{x}"' for x in items[:6]]) + ")")
        skill_query = "(" + " OR ".join(cluster_exprs[:5]) + ")" if cluster_exprs else ""

        recency = '("posted today" OR "last 24 hours" OR "1 day ago" OR "24 hours")'
        if persona.recency_hours > 36 or broadening_level >= 2:
            recency = '("posted" OR "days ago" OR "this week" OR "last week")'

        sites = ""
        if persona.strategy in ("ats_only", "ats_preferred") and persona.site_filters:
            ors = " OR ".join([f"site:{d}" for d in persona.site_filters])
            sites = f"({ors})"

        location = self._resolve_location(state)
        geo = f'("{location}" OR "Remote")'

        fb = (state.refinement_feedback or "").strip()
        extra_neg = " ".join([t for t in fb.split() if t.startswith("-")]) if fb else ""

        q = " ".join([must, skill_query, geo, recency, sites, neg, extra_neg]).strip()
        state.query_modifiers["broadening_level"] = broadening_level
        state.query_modifiers["last_query"] = q
        return q

    def search(self, state: AgentState, limit: int = 25) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        location = self._resolve_location(state)
        country_code = self._resolve_country_code(state)

        for corrective_attempt in range(0, 3):
            query = self.build_query(state)
            for board in ("linkedin", "indeed"):
                results.extend(self._retry_scrape(board, query, attempts=2, timeout_s=30, location=location, country_code=country_code))
            results.extend(self.serper.search(query=f"{query} jobs", num=12))

            deduped = self._dedupe(results, limit=limit)
            if deduped:
                return deduped

            state.query_modifiers["broadening_level"] = int(state.query_modifiers.get("broadening_level") or 0) + 1
            state.log_eval(
                f"[L3 Self-Correction] jobs_raw=0, increasing broadening_level to {state.query_modifiers['broadening_level']} (attempt {corrective_attempt + 1}/3)"
            )

        return self._dedupe(results, limit=limit)

    @staticmethod
    def _dedupe(results: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for r in results:
            url = canonical_url(r.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"title": r.get("title"), "url": url, "snippet": r.get("snippet")})
            if len(out) >= limit:
                break
        return out

    def _retry_scrape(self, board: str, query: str, *, attempts: int, timeout_s: int, location: str, country_code: str) -> List[Dict[str, Any]]:
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                time.sleep(random.uniform(0.4, 1.2) * attempt)
                return self._scrape_board(board, query, timeout_s, location=location, country_code=country_code)
            except Exception as exc:
                last_error = str(exc)
        return [{"title": f"{board} scrape failed", "url": "", "snippet": f"Read Timeout/blocked: {last_error}"}]

    @staticmethod
    def _search_url(board: str, query: str, *, location: str, country_code: str) -> str:
        q = quote_plus(query)
        if board == "linkedin":
            return "https://www.linkedin.com/jobs/search/" f"?keywords={q}&location={quote_plus(location)}&f_TPR=r86400"
        return f"https://www.indeed.com/jobs?q={q}&l={quote_plus(location)}" f"&fromage=1&country={quote_plus(country_code.lower())}"

    def _scrape_board(self, board: str, query: str, timeout_s: int, *, location: str, country_code: str) -> List[Dict[str, Any]]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright

        url = self._search_url(board, query, location=location, country_code=country_code)
        timeout_ms = timeout_s * 1000

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1366, "height": 900},
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9", "Upgrade-Insecure-Requests": "1", "DNT": "1"},
            )
            page = context.new_page()
            try:
                try:
                    from playwright_stealth import stealth_sync

                    stealth_sync(page)
                except Exception:
                    pass
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except PlaywrightTimeoutError as exc:
                browser.close()
                raise TimeoutError(f"{board} L3 Read Timeout after {timeout_s}s") from exc

            page.wait_for_timeout(int(random.uniform(900, 2200)))
            page.mouse.wheel(0, random.randint(500, 1200))
            page.wait_for_timeout(int(random.uniform(700, 1600)))

            anchors = page.eval_on_selector_all(
                "a[href]",
                """
                (els) => els.map((a) => ({
                  title: (a.textContent || '').trim(),
                  url: a.href || '',
                }))
                """,
            )
            browser.close()

        out: List[Dict[str, Any]] = []
        for a in anchors:
            href = str(a.get("url") or "")
            title = str(a.get("title") or "")
            if board == "linkedin" and "linkedin.com/jobs/view" not in href:
                continue
            if board == "indeed" and "/viewjob" not in href:
                continue
            if len(title.strip()) < 3:
                continue
            out.append({"title": title.strip(), "url": href, "snippet": f"source={board}"})
            if len(out) >= 20:
                break
        return out


class GeoFenceManager:
    """Description: Reject only if explicit location metadata outside US.
    Layer: L3
    """

    def filter(self, state: AgentState, jobs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        rejected: List[str] = []
        kept: List[Dict[str, Any]] = []

        for j in jobs:
            url = j.get("url") or ""
            loc = extract_explicit_location(j.get("snippet") or "", j.get("title") or "", "")
            if is_outside_target_geo(url, [state.preferences.location, state.preferences.country], explicit_location=loc or ""):
                rejected.append(f"reject outside target geo: {url} loc='{loc or 'unknown'}'")
                continue

            j["location_hint"] = loc
            kept.append(j)

        return kept, rejected


class ExtractionManager:
    """Description: Uses Jina Reader for full text extraction.
    Layer: L3
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.robots = RobotsGuard(settings)
        self.jina = JinaReader(settings, self.robots)

    def enrich(self, state: AgentState, jobs: List[Dict[str, Any]], max_jobs: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        out: List[Dict[str, Any]] = []
        notes: List[str] = []
        for j in jobs[:max_jobs]:
            url = j.get("url")
            if not url:
                continue
            md, err = self.jina.fetch_markdown(url)
            if err and "robots" in err:
                state.robots_violations.append(f"{url}: {err}")
                continue
            if md:
                j["full_text_md"] = md[:30000]
            if err:
                notes.append(f"{url}: {err}")
            out.append(j)
        return out, notes
