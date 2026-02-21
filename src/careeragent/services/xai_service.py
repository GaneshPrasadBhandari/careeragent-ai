from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from careeragent.orchestration.state import OrchestrationState
from careeragent.services.health_service import get_artifacts_root


class TransparencyMatrixRow(BaseModel):
    """
    Description: One-row transparency explanation of the InterviewChanceScore computation.
    Layer: L9
    Input: weights + components + market factor
    Output: Human-auditable formula breakdown
    """
    model_config = ConfigDict(extra="forbid")

    job_id: str
    role_title: Optional[str] = None
    company: Optional[str] = None

    skill_overlap: float
    experience_alignment: float
    ats_score: float
    market_factor: float

    w_skill: float
    w_exp: float
    w_ats: float

    contrib_skill: float
    contrib_exp: float
    contrib_ats: float
    weighted_sum: float
    final_score: float
    final_percent: float

    notes: List[str] = Field(default_factory=list)


class TransparencyMatrix(BaseModel):
    """
    Description: Transparency matrix for all jobs in a run.
    Layer: L9
    Input: OrchestrationState meta (job_components, weights)
    Output: Matrix JSON stored alongside XAI PDF
    """
    model_config = ConfigDict(extra="forbid")

    run_id: str
    rows: List[TransparencyMatrixRow] = Field(default_factory=list)


class XAIService:
    """
    Description: L9 explainable analytics enhancement for DeepMilestoneReport.
    Layer: L9
    Input: OrchestrationState
    Output: Transparency Matrix JSON + reportlab PDF (preferred) under artifacts/reports/<run_id>/
    """

    def __init__(self, *, artifacts_root: Optional[Path] = None) -> None:
        """
        Description: Initialize XAI service.
        Layer: L0
        Input: artifacts_root optional
        Output: XAIService
        """
        self._root = artifacts_root or get_artifacts_root()

    def build_transparency_matrix(self, *, state: OrchestrationState) -> TransparencyMatrix:
        """
        Description: Build a transparency matrix from state meta.
        Layer: L9
        Input: OrchestrationState
        Output: TransparencyMatrix
        """
        w1 = float(state.meta.get("w1_skill_overlap", 0.45))
        w2 = float(state.meta.get("w2_experience_alignment", 0.35))
        w3 = float(state.meta.get("w3_ats_score", 0.20))

        # Normalize weights defensively (even if upstream already normalized)
        s = (w1 + w2 + w3) or 1.0
        w1, w2, w3 = w1 / s, w2 / s, w3 / s

        comps_map: Dict[str, Any] = state.meta.get("job_components", {}) or {}
        scores_map: Dict[str, Any] = state.meta.get("job_scores", {}) or {}
        meta_map: Dict[str, Any] = state.meta.get("job_meta", {}) or {}

        rows: List[TransparencyMatrixRow] = []
        for job_id, comps in comps_map.items():
            if not isinstance(comps, dict):
                continue

            sk = float(comps.get("skill_overlap", 0.0))
            ex = float(comps.get("experience_alignment", 0.0))
            ats = float(comps.get("ats_score", 0.0))
            mf = float(comps.get("market_competition_factor", 1.0))
            mf = max(1.0, mf)

            contrib_skill = w1 * sk
            contrib_exp = w2 * ex
            contrib_ats = w3 * ats
            weighted_sum = contrib_skill + contrib_exp + contrib_ats
            final = max(0.0, min(1.0, weighted_sum / mf))
            final_pct = round(final * 100.0, 2)

            # If we have a stored score, compare to confirm consistency
            notes: List[str] = []
            if job_id in scores_map:
                try:
                    stored = float(scores_map[job_id])
                    if abs(stored - final) > 1e-6:
                        notes.append(f"Warning: stored score ({stored:.6f}) != recomputed ({final:.6f}).")
                except Exception:
                    pass

            meta = meta_map.get(job_id, {}) if isinstance(meta_map, dict) else {}
            rows.append(
                TransparencyMatrixRow(
                    job_id=str(job_id),
                    role_title=meta.get("role_title"),
                    company=meta.get("company"),
                    skill_overlap=sk,
                    experience_alignment=ex,
                    ats_score=ats,
                    market_factor=mf,
                    w_skill=w1,
                    w_exp=w2,
                    w_ats=w3,
                    contrib_skill=contrib_skill,
                    contrib_exp=contrib_exp,
                    contrib_ats=contrib_ats,
                    weighted_sum=weighted_sum,
                    final_score=final,
                    final_percent=final_pct,
                    notes=notes,
                )
            )

        return TransparencyMatrix(run_id=state.run_id, rows=rows)

    def write_outputs(self, *, state: OrchestrationState, require_reportlab: bool = False) -> Dict[str, str]:
        """
        Description: Write transparency JSON + PDF to artifacts/reports/<run_id>/.
        Layer: L9
        Input: state + require_reportlab
        Output: dict paths {json, pdf}
        """
        out_dir = self._root / "reports" / state.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        matrix = self.build_transparency_matrix(state=state)

        json_path = out_dir / "transparency_matrix.json"
        json_path.write_text(json.dumps(matrix.model_dump(), indent=2), encoding="utf-8")

        pdf_path = out_dir / "xai_transparency_report.pdf"
        self._render_pdf_reportlab(matrix=matrix, out_path=pdf_path, require_reportlab=require_reportlab)

        return {"json": str(json_path), "pdf": str(pdf_path)}

    @staticmethod
    def _render_pdf_reportlab(*, matrix: TransparencyMatrix, out_path: Path, require_reportlab: bool) -> None:
        """
        Description: Render Transparency Matrix into a clear reportlab PDF.
        Layer: L9
        Input: TransparencyMatrix + output path
        Output: PDF file written
        """
        try:
            from reportlab.lib import colors  # type: ignore
            from reportlab.lib.pagesizes import LETTER, landscape  # type: ignore
            from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # type: ignore
        except ModuleNotFoundError as e:
            if require_reportlab:
                raise
            # Soft fallback: write a minimal “pdf-like” file is not acceptable here; instead write TXT next to it.
            txt_path = out_path.with_suffix(".txt")
            lines = ["CareerAgent-AI — XAI Transparency Matrix", f"Run: {matrix.run_id}", ""]
            for r in matrix.rows:
                lines.append(f"{r.job_id}: {r.final_percent:.2f}% = ({r.w_skill:.2f}*{r.skill_overlap:.2f} + {r.w_exp:.2f}*{r.experience_alignment:.2f} + {r.w_ats:.2f}*{r.ats_score:.2f}) / {r.market_factor:.2f}")
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            # Create an empty PDF placeholder with instructions
            out_path.write_text("reportlab not installed. Install with: uv add reportlab", encoding="utf-8")
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(out_path), pagesize=landscape(LETTER), title="XAI Transparency Matrix")
        styles = getSampleStyleSheet()

        story = []
        story.append(Paragraph("CareerAgent-AI — XAI Transparency Matrix", styles["Title"]))
        story.append(Paragraph(f"Run ID: {matrix.run_id}", styles["Normal"]))
        story.append(Spacer(1, 10))

        # Table header + rows
        data = [
            [
                "Job ID",
                "SkillOverlap",
                "ExpAlign",
                "ATS",
                "Market",
                "W_skill (45%)",
                "W_exp (35%)",
                "W_ats (20%)",
                "Contrib_skill",
                "Contrib_exp",
                "Contrib_ats",
                "WeightedSum",
                "FinalScore",
                "Final%",
            ]
        ]

        for r in matrix.rows[:20]:
            data.append(
                [
                    r.job_id,
                    f"{r.skill_overlap:.3f}",
                    f"{r.experience_alignment:.3f}",
                    f"{r.ats_score:.3f}",
                    f"{r.market_factor:.2f}",
                    f"{r.w_skill:.2f}",
                    f"{r.w_exp:.2f}",
                    f"{r.w_ats:.2f}",
                    f"{r.contrib_skill:.3f}",
                    f"{r.contrib_exp:.3f}",
                    f"{r.contrib_ats:.3f}",
                    f"{r.weighted_sum:.3f}",
                    f"{r.final_score:.3f}",
                    f"{r.final_percent:.2f}%",
                ]
            )

        t = Table(data, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

        story.append(
            Paragraph(
                "Formula: FinalScore = (0.45×SkillOverlap + 0.35×ExperienceAlignment + 0.20×ATS) ÷ MarketFactor.",
                styles["Normal"],
            )
        )

        doc.build(story)
