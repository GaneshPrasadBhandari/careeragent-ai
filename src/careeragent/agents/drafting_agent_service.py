from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, ArtifactRef
from careeragent.nlp.skills import compute_jd_alignment, extract_skills, normalize_skill
from careeragent.tools.llm_tools import GeminiClient
from careeragent.tools.web_tools import stable_key


def _write_docx(text: str, path: Path) -> None:
    doc = Document()
    for line in text.splitlines():
        if line.strip().startswith("#"):
            doc.add_heading(line.strip("# ").strip(), level=1)
        elif line.strip() == "":
            doc.add_paragraph("")
        else:
            doc.add_paragraph(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


def _write_pdf(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 50
    for line in text.splitlines():
        if y < 60:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line[:110])
        y -= 14
    c.save()


def ats_format_check_percent(resume_md: str) -> float:
    """ATS format/layout check (structure), independent of JD alignment.

    Description: Validate ATS-safe layout (single column, headings, bullets, no tables).
    Layer: L6
    Input: resume markdown
    Output: percent 0-100
    """

    txt = resume_md or ""
    score = 100.0

    # penalize tables / columns
    if "|" in txt and re.search(r"\n\s*\|", txt):
        score -= 35

    # needs headings
    if not re.search(r"^#\s+", txt, flags=re.M):
        score -= 20

    if not re.search(r"^##\s+", txt, flags=re.M):
        score -= 10

    # needs some bullets
    bullets = len(re.findall(r"^\s*[-•]\s+", txt, flags=re.M))
    if bullets < 6:
        score -= 15

    # avoid very long lines (often indicates copy/paste paragraphs that ATS parses poorly)
    long_lines = sum(1 for ln in txt.splitlines() if len(ln) > 140)
    if long_lines >= 3:
        score -= 10

    return float(max(0.0, min(100.0, score)))


def ats_keyword_match_percent(job_text: str, resume_text: str, *, profile_skills: List[str]) -> float:
    """Entity-based ATS keyword match percent.

    This replaces token-frequency overlap that can collapse to tiny values.

    Description: Entity matching between JD skills and resume skills.
    Layer: L6
    Input: job text + resume text
    Output: percent 0-100
    """

    prof = [normalize_skill(s) for s in (profile_skills or []) if s]
    jd_skills = set(extract_skills(job_text, extra_candidates=prof))
    resume_skills = set(extract_skills(resume_text, extra_candidates=prof))

    if not jd_skills:
        return 0.0

    matched = len(jd_skills & resume_skills)
    return round((matched / max(1, len(jd_skills))) * 100.0, 2)


class DraftingAgentService:
    """Description: Generate ATS-friendly resume + US-style story cover letter.

    Layer: L6
    Input: approved jobs + extracted_profile
    Output: resume_*, cover_* artifacts

    Improvements in this patch:
      - Split ATS checks into:
          1) ATS Format Check (layout)
          2) JD Alignment (entity skills)
      - Compute entity-based ATS keyword match percent (not token frequency)
      - Cover letters become story-based: Action → Result → Challenge.
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.gemini = GeminiClient(settings)

    def generate_for_jobs(self, state: AgentState) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        prof = state.extracted_profile
        profile_skills = list(prof.get("skills") or [])

        for url in state.approved_job_urls[: state.preferences.draft_count]:
            job = next((j for j in state.ranking if str(j.get("url") or "") == url), None)
            if not job:
                continue

            key = stable_key(url)
            jd = (job.get("full_text_md") or job.get("snippet") or "")
            title = job.get("title") or "Target Role"

            resume_md, cover_md = self._gen_with_llm(prof, jd, title)

            # --- Scoring
            layout_pct = ats_format_check_percent(resume_md)

            # JD alignment against *generated resume* text
            align = compute_jd_alignment(jd_text=jd, resume_skills=extract_skills(resume_md, extra_candidates=profile_skills))
            jd_align_pct = align.jd_alignment_percent
            gap_pct = align.missing_skills_gap_percent

            # entity-based keyword match percent
            kw_pct = ats_keyword_match_percent(jd, resume_md, profile_skills=profile_skills)

            job["ats_layout_percent"] = layout_pct
            job["jd_alignment_percent"] = jd_align_pct
            job["missing_skills_gap_percent"] = gap_pct
            job["ats_keyword_match_percent"] = kw_pct
            job["missing_jd_skills"] = align.missing_jd_skills[:40]
            job["matched_jd_skills"] = align.matched_jd_skills[:40]

            run_dir = Path("outputs/runs") / state.run_id
            resume_md_path = run_dir / f"resume_{key}.md"
            cover_md_path = run_dir / f"cover_{key}.md"
            resume_docx = run_dir / f"resume_{key}.docx"
            cover_docx = run_dir / f"cover_{key}.docx"
            resume_pdf = run_dir / f"resume_{key}.pdf"
            cover_pdf = run_dir / f"cover_{key}.pdf"

            run_dir.mkdir(parents=True, exist_ok=True)
            resume_md_path.write_text(resume_md, encoding="utf-8")
            cover_md_path.write_text(cover_md, encoding="utf-8")
            _write_docx(resume_md, resume_docx)
            _write_docx(cover_md, cover_docx)
            _write_pdf(resume_md, resume_pdf)
            _write_pdf(cover_md, cover_pdf)

            state.artifacts[f"resume_{key}"] = ArtifactRef(path=str(resume_md_path), mime="text/markdown")
            state.artifacts[f"cover_{key}"] = ArtifactRef(path=str(cover_md_path), mime="text/markdown")
            state.artifacts[f"resume_{key}_docx"] = ArtifactRef(
                path=str(resume_docx),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            state.artifacts[f"cover_{key}_docx"] = ArtifactRef(
                path=str(cover_docx),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            state.artifacts[f"resume_{key}_pdf"] = ArtifactRef(path=str(resume_pdf), mime="application/pdf")
            state.artifacts[f"cover_{key}_pdf"] = ArtifactRef(path=str(cover_pdf), mime="application/pdf")

            out.append(
                {
                    "job_url": url,
                    "key": key,
                    "ats_layout_percent": layout_pct,
                    "jd_alignment_percent": jd_align_pct,
                    "missing_skills_gap_percent": gap_pct,
                    "ats_keyword_match_percent": kw_pct,
                }
            )

        return out

    def _gen_with_llm(self, profile: Dict[str, Any], jd: str, title: str) -> Tuple[str, str]:
        name = str(profile.get("name") or "Candidate")

        # pick a strong project/experience slice for story fallback
        exp = (profile.get("experience") or [])
        best = exp[0] if exp else {}
        bullets = (best.get("bullets") or [])[:6]

        fallback_resume = (
            "# ATS Resume\n"
            f"## Target Role: {title}\n\n"
            "## Summary\n"
            + str(profile.get("summary") or "")
            + "\n\n## Key Skills\n"
            + ", ".join((profile.get("skills") or [])[:30])
            + "\n\n## Experience\n"
            + "\n".join([f"- {b}" for b in bullets])
        )

        fallback_cover = (
            f"# Cover Letter\n\nDear Hiring Manager,\n\n"
            f"I’m applying for the {title} role. One project I’m proud of is where I {bullets[0] if bullets else 'delivered a production AI solution'} "
            "— the action mattered because it produced measurable results, and it prepared me for the next challenge: scaling and hardening systems under real constraints.\n\n"
            "Action → Result → Challenge:\n"
            f"- **Action:** {bullets[0] if bullets else 'Built an end-to-end AI workflow'}\n"
            f"- **Result:** {bullets[1] if len(bullets) > 1 else 'Improved reliability and delivery speed'}\n"
            f"- **Challenge:** {bullets[2] if len(bullets) > 2 else 'Owned ambiguity and stakeholder alignment'}\n\n"
            "I’d bring the same execution discipline to your team — shipping clean architecture, measurable outcomes, and safe automation.\n\n"
            f"Sincerely,\n{name}\n"
        )

        if not self.s.GEMINI_API_KEY:
            return fallback_resume, fallback_cover

        # US-style story cover letter prompt
        prompt = (
            "You are a senior US resume + cover letter writer for a Staff/Senior AI/GenAI Solution Architect.\n"
            "You MUST follow these rules:\n"
            "- Resume must be ATS-friendly MARKDOWN (no tables, no columns, no images).\n"
            "- Cover letter must be story-based (Action → Result → Challenge).\n"
            "- Identify ONE major project/initiative from the candidate profile experience.\n"
            "- Infer the job's biggest challenge from the JD (what they are struggling to ship).\n"
            "- Map the selected project to that challenge with clear cause-effect logic.\n"
            "- Use ONLY facts from the candidate profile. Do NOT invent employers, degrees, or metrics. If a metric isn't provided, say 'improved' without numbers.\n"
            "Return STRICT JSON: {resume_md: string, cover_md: string}.\n\n"
            f"CANDIDATE_PROFILE_JSON: {profile}\n\n"
            f"JOB_TITLE: {title}\n"
            f"JOB_DESCRIPTION:\n{jd[:9000]}\n"
        )

        j = self.gemini.generate_json(prompt, temperature=0.2, max_tokens=1500)
        if not isinstance(j, dict):
            return fallback_resume, fallback_cover

        resume_md = str(j.get("resume_md") or "").strip() or fallback_resume
        cover_md = str(j.get("cover_md") or "").strip() or fallback_cover
        return resume_md, cover_md
