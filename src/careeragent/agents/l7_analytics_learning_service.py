from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from careeragent.core.mcp_client import MCPClient, sqlite_path_from_database_url
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, ArtifactRef
from careeragent.tools.llm_tools import GeminiClient


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DashboardManager:
    """Description: Track applications and interview status in SQLite.
    Layer: L7
    Input: approved jobs
    Output: job_tracker rows
    """

    def __init__(self, settings: Settings, mcp: MCPClient) -> None:
        self.s = settings
        self.mcp = mcp

    def record_approved(self, state: AgentState) -> None:
        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        for url in state.approved_job_urls:
            job = next((j for j in state.ranking if str(j.get("url") or "") == url), {})
            company = str(job.get("company") or job.get("source") or "")
            priority = "high" if float(job.get("overall_match_percent") or 0.0) >= 75 else "med"
            # Create if missing, then update status.
            self.mcp.sqlite_exec(
                db_path,
                "INSERT OR IGNORE INTO job_tracker(run_id, applied_date, company, job_url, priority, interview_status) VALUES(?,?,?,?,?,?)",
                (state.run_id, utc_now_iso(), company, url, priority, "drafted"),
            )
            self.mcp.sqlite_exec(
                db_path,
                "UPDATE job_tracker SET interview_status=? WHERE run_id=? AND job_url=?",
                ("drafted", state.run_id, url),
            )

    def record_shortlist(self, state: AgentState, *, status: str = "shortlisted", top_n: int = 8) -> None:
        """Record a shortlist snapshot even if the run pauses.

        Description: Ensures the dashboard has a row even when HITL pauses at L5.
        Layer: L7
        Input: state.ranking
        Output: job_tracker rows (upsert)
        """

        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        for job in (state.ranking or [])[: max(1, int(top_n))]:
            url = str(job.get("url") or job.get("job_id") or "")
            if not url:
                continue
            company = str(job.get("company") or job.get("source") or "")
            priority = "high" if float(job.get("overall_match_percent") or 0.0) >= 75 else "med"
            self.mcp.sqlite_exec(
                db_path,
                "INSERT OR IGNORE INTO job_tracker(run_id, applied_date, company, job_url, priority, interview_status) VALUES(?,?,?,?,?,?)",
                (state.run_id, utc_now_iso(), company, url, priority, status),
            )
            self.mcp.sqlite_exec(
                db_path,
                "UPDATE job_tracker SET interview_status=? WHERE run_id=? AND job_url=?",
                (status, state.run_id, url),
            )


class SelfLearningAgent:
    """Description: Learn from failures to refine future searches.
    Layer: L7
    Input: evaluation failures and retry reasons
    Output: learning_memory table records
    """

    def __init__(self, settings: Settings, mcp: MCPClient) -> None:
        self.s = settings
        self.mcp = mcp

    def ingest_failure(self, state: AgentState, *, user_key: str = "default") -> None:
        db_path = sqlite_path_from_database_url(self.s.DATABASE_URL)
        signal = "retry_loop" if state.retry_count > 0 else "run"
        payload = {
            "run_id": state.run_id,
            "evaluation": state.evaluation,
            "last_query": state.query_modifiers.get("last_query"),
            "active_persona": state.active_persona_id,
            "reasons": state.evaluation_logs[-10:],
        }
        self.mcp.sqlite_exec(
            db_path,
            "INSERT INTO learning_memory(user_key, signal, payload_json, created_at) VALUES(?,?,?,?)",
            (user_key, signal, json.dumps(payload), utc_now_iso()),
        )


class CareerCoach:
    """Description: Generate 6-month roadmap based on gaps.
    Layer: L7
    Input: aggregated missing skills
    Output: roadmap markdown
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.gemini = GeminiClient(settings)

    def build_roadmap(self, state: AgentState) -> str:
        # aggregate missing skills across ranking
        missing: Dict[str, int] = {}
        for j in state.ranking[:25]:
            # v2 matcher uses missing_jd_skills (JD skills not in resume)
            for s in (j.get("missing_jd_skills") or j.get("missing_skills") or [])[:15]:
                missing[str(s)] = missing.get(str(s), 0) + 1
        top = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:18]
        gaps = [k for (k, _) in top]

        fallback = "# 6-Month Upskilling Roadmap\n\n" + "\n".join([f"- Focus: {g}" for g in gaps])

        if not self.s.GEMINI_API_KEY:
            return fallback

        prompt = (
            "Create a 6-month upskilling roadmap for the candidate based on skill gaps.\n"
            "Return MARKDOWN with month-by-month plan (Months 1-6), weekly themes, hands-on projects, and interview prep.\n"
            "Do not invent degrees/employers; use the gap list.\n\n"
            f"CANDIDATE_PROFILE_JSON: {state.extracted_profile}\n\n"
            f"GAPS: {gaps}\n"
        )
        text = self.gemini.generate_text(prompt, temperature=0.35, max_tokens=1200)
        return text or fallback

    def save_roadmap(self, state: AgentState, md: str) -> None:
        run_dir = Path("outputs/runs") / state.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "career_roadmap.md"
        path.write_text(md, encoding="utf-8")
        state.artifacts["career_roadmap"] = ArtifactRef(path=str(path), mime="text/markdown")
