from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from careeragent.core.state import AgentState
from careeragent.services.health_service import get_artifacts_root


class DeepMilestoneMatchExplain(BaseModel):
    """
    Description: XAI explanation for a single match score (why X%).
    Layer: L9
    Input: Match components + weights + market factor
    Output: Human-readable explanation bullets
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    role_title: Optional[str] = None
    company: Optional[str] = None

    interview_chance_score: float
    overall_match_percent: float

    weights: Dict[str, float] = Field(default_factory=dict)
    components: Dict[str, float] = Field(default_factory=dict)
    explanation: List[str] = Field(default_factory=list)


class DeepMilestoneReport(BaseModel):
    """
    Description: Deep milestone report: XAI + market trends + security + quota + outcomes.
    Layer: L9
    Input: AgentState (audit trail)
    Output: JSON + PDF report artifacts
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_status: str

    match_explanations: List[DeepMilestoneMatchExplain] = Field(default_factory=list)
    market_trends: Dict[str, Any] = Field(default_factory=dict)
    security_compliance: Dict[str, Any] = Field(default_factory=dict)
    quota_summary: Dict[str, Any] = Field(default_factory=dict)
    outcome_summary: Dict[str, Any] = Field(default_factory=dict)


class DeepAnalyticsService:
    """
    Description: Generates a Deep Milestone Report (JSON + PDF) with XAI and compliance signals.
    Layer: L9
    Input: AgentState
    Output: Writes artifacts under src/careeragent/artifacts/reports/<run_id>/
    """

    def __init__(self, *, artifacts_root: Optional[Path] = None) -> None:
        """
        Description: Initialize deep analytics service.
        Layer: L0
        Input: Optional artifacts_root
        Output: DeepAnalyticsService
        """
        self._root = artifacts_root or get_artifacts_root()

    def generate(self, *, state: AgentState) -> Dict[str, str]:
        """
        Description: Build report and write JSON + PDF artifacts.
        Layer: L9
        Input: AgentState
        Output: dict with artifact paths
        """
        report = self._build_report(state=state)
        out_dir = self._root / "reports" / state.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "deep_milestone_report.json"
        json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

        pdf_path = out_dir / "deep_milestone_report.pdf"
        self._write_pdf(report=report, out_path=pdf_path)

        return {"json": str(json_path), "pdf": str(pdf_path)}

    def _build_report(self, *, state: AgentState) -> DeepMilestoneReport:
        """
        Description: Construct report fields from AgentState.
        Layer: L9
        Input: AgentState
        Output: DeepMilestoneReport
        """
        # XAI: derive explanations from match report artifacts if present in state.meta['job_scores'] and artifacts
        job_scores = state.meta.get("job_scores", {}) or {}
        explanations: List[DeepMilestoneMatchExplain] = []

        # Optional: if matcher stored per-job components in meta, use them; else fallback to what we can infer.
        job_components = state.meta.get("job_components", {}) or {}

        weights = {
            "w1_skill_overlap": float(state.meta.get("w1_skill_overlap", 0.45)),
            "w2_experience_alignment": float(state.meta.get("w2_experience_alignment", 0.35)),
            "w3_ats_score": float(state.meta.get("w3_ats_score", 0.20)),
        }

        for job_id, score in job_scores.items():
            try:
                s = float(score)
            except Exception:
                s = 0.0
            comps = job_components.get(job_id, {})
            overall = round(s * 100.0, 2)

            bullets = [
                f"Score = (0.45×SkillOverlap + 0.35×ExperienceAlignment + 0.20×ATS) ÷ MarketFactor.",
            ]
            if comps:
                bullets.append(
                    f"Components: SkillOverlap={comps.get('skill_overlap', 'n/a')}, "
                    f"ExperienceAlignment={comps.get('experience_alignment', 'n/a')}, "
                    f"ATS={comps.get('ats_score', 'n/a')}, MarketFactor={comps.get('market_competition_factor', 'n/a')}."
                )
            bullets.append("Primary drivers are the largest weighted components (skills, then experience alignment).")

            explanations.append(
                DeepMilestoneMatchExplain(
                    job_id=str(job_id),
                    interview_chance_score=s,
                    overall_match_percent=overall,
                    weights=weights,
                    components={k: float(v) for k, v in comps.items()} if isinstance(comps, dict) else {},
                    explanation=bullets,
                )
            )

        # Market trends: top skills seen in target_role_keywords or match gaps (best-effort)
        kw = state.meta.get("target_role_keywords", []) or []
        missing = []
        for ex in state.meta.get("match_gaps", []) or []:
            if isinstance(ex, str):
                missing.append(ex.lower())

        trend_counter = Counter([str(x).lower() for x in kw] + missing)
        top = trend_counter.most_common(12)

        market_trends = {
            "top_keywords": [{"keyword": k, "count": c} for k, c in top],
            "note": "Keyword counts are derived from job requirements inputs and observed gaps.",
        }

        # Security compliance summary
        sec_events = state.meta.get("security_events", []) or []
        sec_counter = Counter([e.get("type") for e in sec_events if isinstance(e, dict)])
        security_compliance = {
            "security_events_total": len(sec_events),
            "security_event_types": dict(sec_counter),
            "run_failure_code": state.meta.get("run_failure_code"),
        }

        # Quota summary (if present)
        quota = state.meta.get("quota_snapshot") or {}
        quota_summary = quota if isinstance(quota, dict) else {"note": "No quota snapshot found."}

        # Outcome summary (from status updates if any)
        status_updates = state.meta.get("status_updates", []) or []
        outcome_summary = {"status_updates_total": len(status_updates)}
        if status_updates:
            last = status_updates[-1]
            outcome_summary["latest"] = last

        return DeepMilestoneReport(
            run_id=state.run_id,
            run_status=state.status,
            match_explanations=explanations,
            market_trends=market_trends,
            security_compliance=security_compliance,
            quota_summary=quota_summary,
            outcome_summary=outcome_summary,
        )

    @staticmethod
    def _write_pdf(*, report: DeepMilestoneReport, out_path: Path) -> None:
        """
        Description: Render a simple PDF report for sharing and governance review.
        Layer: L9
        Input: DeepMilestoneReport + output path
        Output: PDF file written to disk
        """
        # reportlab is available in your environment
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(out_path), pagesize=LETTER)
        width, height = LETTER

        y = height - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "CareerAgent-AI — Deep Milestone Report")
        y -= 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Run ID: {report.run_id}")
        y -= 14
        c.drawString(50, y, f"Run Status: {report.run_status}")
        y -= 22

        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Explainable Match Scores")
        y -= 16
        c.setFont("Helvetica", 9)

        if not report.match_explanations:
            c.drawString(50, y, "No match explanations found in state.")
            y -= 14
        else:
            for ex in report.match_explanations[:6]:
                c.drawString(50, y, f"- {ex.job_id}: {ex.overall_match_percent:.2f}% (score={ex.interview_chance_score:.3f})")
                y -= 12
                for b in ex.explanation[:2]:
                    c.drawString(65, y, f"• {b[:110]}")
                    y -= 12
                y -= 4
                if y < 120:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 9)

        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Market Trends (Top Keywords)")
        y -= 16
        c.setFont("Helvetica", 9)
        for row in report.market_trends.get("top_keywords", [])[:10]:
            c.drawString(50, y, f"- {row['keyword']}: {row['count']}")
            y -= 12
            if y < 120:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 9)

        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Security & Compliance")
        y -= 16
        c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Security events: {report.security_compliance.get('security_events_total', 0)}")
        y -= 12
        c.drawString(50, y, f"Event types: {json.dumps(report.security_compliance.get('security_event_types', {}))[:110]}")
        y -= 12
        c.drawString(50, y, f"Run failure code: {report.security_compliance.get('run_failure_code')}")
        y -= 18

        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Outcomes")
        y -= 16
        c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Status updates total: {report.outcome_summary.get('status_updates_total', 0)}")
        y -= 12
        if report.outcome_summary.get("latest"):
            c.drawString(50, y, f"Latest: {json.dumps(report.outcome_summary.get('latest'))[:110]}")
            y -= 12

        c.save()
