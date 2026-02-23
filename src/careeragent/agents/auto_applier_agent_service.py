from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState
from careeragent.tools.web_tools import RobotsGuard


@dataclass
class ApplyResult:
    job_url: str
    status: str
    message: str


class AutoApplierAgentService:
    """Description: Playwright-based auto applier (safe demo mode).
    Layer: L6
    Input: approved jobs + tailored resume paths
    Output: apply results

    Safety:
      - Does NOT bypass CAPTCHAs.
      - Stops at final submit unless user explicitly enables submit.
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.robots = RobotsGuard(settings)

    def apply(self, state: AgentState, *, dry_run: bool = True) -> List[ApplyResult]:
        results: List[ApplyResult] = []

        for url in state.approved_job_urls:
            dec = self.robots.allowed(url)
            if not dec.allowed:
                results.append(ApplyResult(job_url=url, status="skipped", message=dec.reason))
                continue

            try:
                # Optional dependency
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, timeout=45_000)
                    # Demo behavior: just confirm page is reachable
                    title = page.title()
                    browser.close()
                results.append(ApplyResult(job_url=url, status="dry_run", message=f"Opened page: {title[:80]}") )
            except Exception as e:
                # If playwright isn't installed or fails, still return a safe plan.
                results.append(ApplyResult(job_url=url, status="skipped", message=f"Playwright not available or failed: {e}"))

        return results
