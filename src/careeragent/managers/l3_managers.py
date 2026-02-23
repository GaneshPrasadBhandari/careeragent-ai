from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState
from careeragent.tools.web_tools import (
    JinaReader,
    RobotsGuard,
    SerperClient,
    TavilyClient,
    canonical_url,
    domain_is_india,
    extract_explicit_location,
    is_non_us_location,
)


class LeadScout:
    """Description: LeadScout generates and executes search personas.
    Layer: L3
    Input: AgentState.search_personas
    Output: jobs_raw list
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.serper = SerperClient(settings)
        self.tavily = TavilyClient(settings)

    def build_query(self, state: AgentState) -> str:
        persona = next((p for p in state.search_personas if p.persona_id == state.active_persona_id), None)
        if not persona:
            persona = state.search_personas[0]
            state.active_persona_id = persona.persona_id

        must = " ".join([f'"{x}"' for x in persona.must_include if x])
        neg = " ".join([f"-{x}" for x in persona.negative_terms if x])

        recency = ""
        if persona.recency_hours <= 36:
            recency = '("posted today" OR "last 24 hours" OR "1 day ago" OR "24 hours")'
        else:
            recency = '("posted" OR "days ago" OR "this week")'

        sites = ""
        if persona.strategy in ("ats_only", "ats_preferred") and persona.site_filters:
            # ATS-preferred: add sites as OR terms but do not hard filter unless ats_only
            ors = " OR ".join([f"site:{d}" for d in persona.site_filters])
            sites = f"({ors})"

        geo = ""
        if state.preferences.country.upper() == "US":
            geo = '("United States" OR "USA" OR "Remote")'

        # refinement feedback injection
        fb = (state.refinement_feedback or "").strip()
        extra_neg = ""
        if fb:
            # crude extraction: any '-X' terms already present
            extra_neg = " ".join([t for t in fb.split() if t.startswith("-")])

        q = " ".join([must, geo, recency, sites, neg, extra_neg]).strip()
        return q

    def search(self, state: AgentState, limit: int = 25) -> List[Dict[str, Any]]:
        query = self.build_query(state)
        state.query_modifiers["last_query"] = query

        results: List[Dict[str, Any]] = []
        # Prefer Serper, fall back Tavily
        if self.s.SERPER_API_KEY:
            results = self.serper.search(query, num=min(limit, 20))
        if not results and self.s.TAVILY_API_KEY:
            results = self.tavily.search(query, max_results=min(limit, 20))

        # dedupe and clean
        seen = set()
        out: List[Dict[str, Any]] = []
        for r in results:
            url = canonical_url(r.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"title": r.get("title"), "url": url, "snippet": r.get("snippet")})
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
            if domain_is_india(url):
                rejected.append(f"reject domain india: {url}")
                continue

            # Only use explicit metadata: snippet/title/head; NOT full body mentions.
            loc = extract_explicit_location(j.get("snippet") or "", j.get("title") or "", "")
            if state.preferences.country.upper() == "US" and loc and is_non_us_location(loc):
                rejected.append(f"reject non-us location '{loc}': {url}")
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
