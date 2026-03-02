from __future__ import annotations

from typing import Any, Dict, List

from careeragent.core.state import AgentState


class FunnelAuditService:
    """Build discovered→ranked→approved→attempted→submitted funnel and blocker taxonomy."""

    @staticmethod
    def build(state: AgentState) -> Dict[str, Any]:
        discovered = len(state.jobs_raw or [])
        ranked = len(state.ranking or state.jobs_scored or [])
        approved = len(state.approved_job_urls or [])

        attempts: List[Dict[str, Any]] = list(state.meta.get("apply_attempts") or [])
        attempted = len(attempts)

        submissions = state.meta.get("submissions") or {}
        submitted = len(submissions) if isinstance(submissions, dict) else 0
        if submitted == 0:
            submitted = sum(1 for a in attempts if str(a.get("status") or "") in {"submitted", "mapped_pending_submit"})

        blockers: Dict[str, int] = {}
        for a in attempts:
            status = str(a.get("status") or "unknown")
            reason = str(a.get("reason") or a.get("message") or "")
            bucket = FunnelAuditService._bucket_blocker(status, reason)
            blockers[bucket] = blockers.get(bucket, 0) + 1

        rates = {
            "discover_to_rank": FunnelAuditService._pct(ranked, discovered),
            "rank_to_approved": FunnelAuditService._pct(approved, ranked),
            "approved_to_attempted": FunnelAuditService._pct(attempted, approved),
            "attempted_to_submitted": FunnelAuditService._pct(submitted, attempted),
            "discover_to_submitted": FunnelAuditService._pct(submitted, discovered),
        }

        return {
            "discovered": discovered,
            "ranked": ranked,
            "approved": approved,
            "attempted": attempted,
            "submitted": submitted,
            "conversion_rates": rates,
            "blocker_taxonomy": blockers,
        }

    @staticmethod
    def _pct(n: int, d: int) -> float:
        if d <= 0:
            return 0.0
        return round((float(n) / float(d)) * 100.0, 2)

    @staticmethod
    def _bucket_blocker(status: str, reason: str) -> str:
        s = (status or "").lower()
        r = (reason or "").lower()
        if "captcha" in r or "cloudflare" in r or "human" in r:
            return "anti_bot_captcha"
        if "robots" in r:
            return "robots_policy"
        if "hitl" in s or "approval" in r:
            return "hitl_gate"
        if "low_score" in s or "score" in r:
            return "score_threshold"
        if "missing" in r or "field" in r:
            return "profile_data_gap"
        if "closed" in r or "expired" in r or "no longer accepting" in r:
            return "job_unavailable"
        if "timeout" in r or "failed" in r or "error" in r:
            return "site_or_tool_failure"
        if s in {"submitted", "mapped_pending_submit", "dry_run_mapped"}:
            return "success"
        return "other"
