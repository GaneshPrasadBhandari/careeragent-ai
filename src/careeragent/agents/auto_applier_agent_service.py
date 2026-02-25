from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState
from careeragent.services.notification_service import NotificationService
from careeragent.tools.web_tools import RobotsGuard


@dataclass
class ApplyResult:
    job_url: str
    status: str
    message: str


class AutoApplierAgentService:
    """Description: Playwright-based auto applier for common ATS flows.
    Layer: L7
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.robots = RobotsGuard(settings)
        self.notify = NotificationService(settings=settings)

    def _human_type(self, page: Any, selector: str, text: str) -> bool:
        if not text:
            return False
        try:
            page.click(selector, timeout=2_000)
            page.type(selector, text, delay=random.randint(50, 150))
            time.sleep(random.uniform(0.2, 0.8))
            return True
        except Exception:
            return False

    def _map_common_fields(self, page: Any, profile: Dict[str, Any], url: str) -> List[str]:
        missing: List[str] = []
        contact = profile.get("contact", {}) or {}
        exp = profile.get("experience", []) or []
        skills = profile.get("skills", []) or []

        name = str(profile.get("name") or "")
        email = str(contact.get("email") or "")
        phone = str(contact.get("phone") or "")
        years = str(profile.get("years_experience") or (len(exp) and len(exp)) or "")
        skills_text = ", ".join([str(s) for s in skills[:20]])

        mapped = {
            "input[name*='name' i], input[id*='name' i]": name,
            "input[type='email'], input[name*='email' i]": email,
            "input[type='tel'], input[name*='phone' i]": phone,
            "textarea[name*='skill' i], input[name*='skill' i]": skills_text,
            "input[name*='experience' i], textarea[name*='experience' i]": years,
        }

        for sel, value in mapped.items():
            if not value:
                missing.append(sel)
                continue
            self._human_type(page, sel, value)

        if missing:
            self.notify.send_alert(
                message=f"HITL needed for {url}. Missing fields in profile for selectors: {missing[:4]}",
                title="CareerAgent HITL Required",
                priority="high",
            )
        return missing

    def apply(self, state: AgentState, *, dry_run: bool = True) -> List[ApplyResult]:
        results: List[ApplyResult] = []
        profile = state.extracted_profile or {}

        for url in state.approved_job_urls:
            dec = self.robots.allowed(url)
            if not dec.allowed:
                results.append(ApplyResult(job_url=url, status="skipped", message=dec.reason))
                continue

            ats = "generic"
            lu = url.lower()
            if "greenhouse" in lu:
                ats = "greenhouse"
            elif "workday" in lu:
                ats = "workday"
            elif "lever" in lu:
                ats = "lever"

            try:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, timeout=45_000)
                    missing = self._map_common_fields(page, profile, url)

                    if not dry_run and not missing:
                        self.notify.send_alert(
                            message=f"Final submit pending confirmation for {url}",
                            title="CareerAgent Final Submission Gate",
                            priority="high",
                        )

                    browser.close()
                status = "dry_run_mapped" if dry_run else "mapped_pending_submit"
                results.append(ApplyResult(job_url=url, status=status, message=f"Mapped fields for {ats}"))
            except Exception as e:
                results.append(ApplyResult(job_url=url, status="skipped", message=f"Playwright not available or failed: {e}"))

        return results
