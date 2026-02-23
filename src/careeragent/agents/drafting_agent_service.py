from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, ArtifactRef
from careeragent.tools.llm_tools import GeminiClient
from careeragent.tools.web_tools import stable_key


def _keyword_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9+.#]", (text or "").lower()) if t and len(t) > 2]


def ats_keyword_match(job_text: str, resume_text: str) -> float:
    """Description: ATS keyword match ratio.
    Layer: L6
    Input: job text + resume text
    Output: ratio 0-1
    """
    jd = set(_keyword_tokens(job_text))
    rs = set(_keyword_tokens(resume_text))
    jd_focus = set([t for t in jd if len(t) >= 4])
    if not jd_focus:
        return 0.0
    return len(jd_focus & rs) / max(1, len(jd_focus))


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


class DraftingAgentService:
    """Description: Generate ATS-friendly resume + cover letter.
    Layer: L6
    Input: approved jobs + extracted_profile
    Output: resume_*, cover_* artifacts
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.gemini = GeminiClient(settings)

    def generate_for_jobs(self, state: AgentState) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        prof = state.extracted_profile

        for url in state.approved_job_urls[: state.preferences.draft_count]:
            job = next((j for j in state.ranking if str(j.get("url") or "") == url), None)
            if not job:
                continue

            key = stable_key(url)
            jd = (job.get("full_text_md") or job.get("snippet") or "")
            title = job.get("title") or "Target Role"

            resume_md, cover_md = self._gen_with_llm(prof, jd, title)

            kw = ats_keyword_match(jd, resume_md)
            job["ats_keyword_match"] = round(kw, 3)

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

            out.append({"job_url": url, "key": key, "ats_keyword_match": kw})

        return out

    def _gen_with_llm(self, profile: Dict[str, Any], jd: str, title: str) -> Tuple[str, str]:
        fallback_resume = (
            "# ATS Resume\n"
            f"## Target Role: {title}\n\n"
            "## Summary\n"
            + str(profile.get("summary") or "")
            + "\n\n## Key Skills\n"
            + ", ".join((profile.get("skills") or [])[:30])
            + "\n\n## Experience\n"
            + "\n".join([f"- {b}" for e in (profile.get("experience") or [])[:3] for b in (e.get("bullets") or [])[:4]])
        )

        fallback_cover = (
            f"# Cover Letter\n\nDear Hiring Manager,\n\n"
            f"I’m applying for the {title} role. I bring experience in AI/ML, GenAI, cloud architecture, and production MLOps. "
            "I’m excited to contribute with proven delivery across enterprise projects.\n\n"
            "Sincerely,\n"
            + str(profile.get("name") or "")
        )

        if not self.s.GEMINI_API_KEY:
            return fallback_resume, fallback_cover

        prompt = (
            "You are an expert US resume writer. Produce an ATS-friendly resume in MARKDOWN (no tables, no images).\n"
            "Rules:\n"
            "- Use US ATS format: clear headings, bullets, no columns, no graphics.\n"
            "- Use ONLY facts present in the candidate profile JSON; do not invent employers, degrees, or metrics.\n"
            "- Tailor keywords to the job description.\n"
            "Return STRICT JSON: {resume_md: string, cover_md: string}.\n\n"
            f"CANDIDATE_PROFILE_JSON: {profile}\n\nJOB_DESCRIPTION:\n{jd[:9000]}\n"
        )
        j = self.gemini.generate_json(prompt, temperature=0.25, max_tokens=1400)
        if not isinstance(j, dict):
            return fallback_resume, fallback_cover

        resume_md = str(j.get("resume_md") or "").strip() or fallback_resume
        cover_md = str(j.get("cover_md") or "").strip() or fallback_cover
        return resume_md, cover_md
