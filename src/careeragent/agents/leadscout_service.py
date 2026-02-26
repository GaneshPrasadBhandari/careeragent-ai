from __future__ import annotations

import random
import time
from typing import Any, Dict, List
from urllib.parse import quote_plus

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState


class LeadScoutService:
    """Playwright-backed L3 discovery with human-mimicry and retry logic."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings

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

    def build_query(self, state: AgentState) -> str:
        persona = next((p for p in state.search_personas if p.persona_id == state.active_persona_id), None)
        if not persona:
            persona = state.search_personas[0]
            state.active_persona_id = persona.persona_id

        must = " ".join([f'"{x}"' for x in persona.must_include if x])
        neg = " ".join([f"-{x}" for x in persona.negative_terms if x])
        location = self._resolve_location(state)
        geo = f'"{location}" OR Remote'
        return " ".join([must, geo, neg]).strip()

    def search(self, state: AgentState, limit: int = 25) -> List[Dict[str, Any]]:
        query = self.build_query(state)
        state.query_modifiers["last_query"] = query

        aggregated: List[Dict[str, Any]] = []
        location = self._resolve_location(state)
        country_code = self._resolve_country_code(state)
        for board in ("linkedin", "indeed"):
            board_results = self._retry_scrape(board, query, timeout_s=30, attempts=3, location=location, country_code=country_code)
            aggregated.extend(board_results)

        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in aggregated:
            url = str(item.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _retry_scrape(self, board: str, query: str, timeout_s: int, attempts: int, *, location: str, country_code: str) -> List[Dict[str, Any]]:
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                jitter = random.uniform(0.4, 1.3) * attempt
                time.sleep(jitter)
                return self._scrape_board(board=board, query=query, timeout_s=timeout_s, location=location, country_code=country_code)
            except Exception as exc:
                last_error = str(exc)
        return [{"title": f"{board} scrape failed", "url": "", "snippet": f"Read Timeout/blocked: {last_error}"}]

    def _search_url(self, board: str, query: str, *, location: str, country_code: str) -> str:
        q = quote_plus(query)
        if board == "linkedin":
            return (
                "https://www.linkedin.com/jobs/search/"
                f"?keywords={q}&location={quote_plus(location)}&f_TPR=r86400"
            )
        return (
            f"https://www.indeed.com/jobs?q={q}&l={quote_plus(location)}"
            f"&fromage=1&country={quote_plus(country_code.lower())}"
        )

    def _scrape_board(self, board: str, query: str, timeout_s: int, *, location: str, country_code: str) -> List[Dict[str, Any]]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright

        timeout_ms = timeout_s * 1000
        url = self._search_url(board, query, location=location, country_code=country_code)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1366, "height": 900},
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Upgrade-Insecure-Requests": "1",
                    "DNT": "1",
                },
            )
            page = context.new_page()

            try:
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

        results: List[Dict[str, Any]] = []
        for a in anchors:
            href = str(a.get("url") or "")
            title = str(a.get("title") or "").strip()
            if board == "linkedin" and "linkedin.com/jobs/view" not in href:
                continue
            if board == "indeed" and "/viewjob" not in href:
                continue
            if len(title) < 3:
                continue
            results.append({"title": title, "url": href, "snippet": f"source={board}"})
            if len(results) >= 20:
                break
        return results
