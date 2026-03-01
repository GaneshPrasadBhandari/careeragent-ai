from __future__ import annotations

import random
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState
from careeragent.services.notification_service import NotificationService
from careeragent.tools.web_tools import RobotsGuard, apply_playwright_stealth


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
        ranking_map = {str(j.get("url") or ""): j for j in (state.ranking or [])}
        state.meta.setdefault("apply_attempts", [])

        for url in state.approved_job_urls:
            score = float((ranking_map.get(url) or {}).get("phase2_score") or (ranking_map.get(url) or {}).get("match_score") or 0.0)
            if score < 0.65:
                msg = f"score={score:.2f} below 0.65"
                results.append(ApplyResult(job_url=url, status="skipped_low_score", message=msg))
                state.meta["apply_attempts"].append({"job_url": url, "status": "skipped_low_score", "reason": msg, "source": urlparse(url).netloc})
                continue
            if 0.65 < score < 0.85:
                self.notify.send_alert(
                    message=f"Run {state.run_id}: HITL go/no-go required for {url} (score={score:.2f}).",
                    title="CareerAgent Mid-Score Approval Needed",
                    priority="high",
                )
                msg = f"score={score:.2f} requires approval"
                results.append(ApplyResult(job_url=url, status="hitl_required", message=msg))
                state.meta["apply_attempts"].append({"job_url": url, "status": "hitl_required", "reason": msg, "source": urlparse(url).netloc})
                continue

            dec = self.robots.allowed(url)
            if not dec.allowed:
                results.append(ApplyResult(job_url=url, status="skipped", message=dec.reason))
                state.meta["apply_attempts"].append({"job_url": url, "status": "skipped", "reason": dec.reason, "source": urlparse(url).netloc})
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
                    apply_playwright_stealth(page)
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
                msg = f"Mapped fields for {ats}" if not missing else f"Mapped with missing fields for {ats}"
                results.append(ApplyResult(job_url=url, status=status, message=msg))
                state.meta["apply_attempts"].append({"job_url": url, "status": status, "reason": msg, "source": urlparse(url).netloc, "missing_fields": missing[:8]})
            except Exception as e:
                msg = f"Playwright not available or failed: {e}"
                results.append(ApplyResult(job_url=url, status="skipped", message=msg))
                state.meta["apply_attempts"].append({"job_url": url, "status": "skipped", "reason": msg, "source": urlparse(url).netloc})

        return results
