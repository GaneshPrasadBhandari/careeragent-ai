from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

try:
    from docx import Document
except Exception:  # pragma: no cover
    Document = None

from pydantic import BaseModel, Field

from careeragent.core.settings import Settings
from careeragent.tools.llm_tools import GeminiClient


class ContactModel(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    linkedin: Optional[str] = None
    github: Optional[str] = None


class ExperienceModel(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    years: float = 0.0
    bullets: List[str] = Field(default_factory=list)


class EducationModel(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    graduation_year: Optional[str] = None


class ExtractedResume(BaseModel):
    name: Optional[str] = None
    contact: ContactModel = Field(default_factory=ContactModel)
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceModel] = Field(default_factory=list)
    education: List[EducationModel] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    total_years_experience: float = 0.0
    low_confidence: bool = False
    evaluation_score: float = 0.0


class SkillExtractionValidator(BaseModel):
    skills: List[str] = Field(default_factory=list)


def _rx_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    return m.group(0) if m else None


def _rx_phone(text: str) -> Optional[str]:
    m = re.search(r"(\+?\d{1,3}[-\s]?)?(\(?\d{3}\)?[-\s]?)\d{3}[-\s]?\d{4}", text or "")
    return m.group(0).strip() if m else None


def _guess_name(text: str) -> Optional[str]:
    for line in (text or "").splitlines()[:8]:
        s = line.strip()
        if not s:
            continue
        if any(k in s.lower() for k in ["@", "linkedin", "github", "http"]):
            continue
        if 1 <= len(s.split()) <= 5 and len(s) <= 60:
            return s
    return None


def _extract_docx_text(file_bytes: bytes) -> str:
    if Document is None:
        return ""
    doc = Document(BytesIO(file_bytes))
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip())


def _parse_skills_phase1(text: str) -> List[str]:
    lexicon = [
        "Python", "AWS", "Azure", "GCP", "Machine Learning", "Deep Learning", "NLP",
        "GenAI", "LLM", "RAG", "LangChain", "LangGraph", "MLOps", "TensorFlow",
        "PyTorch", "SQL", "FastAPI", "Docker", "Kubernetes", "Spark", "Databricks",
        "Solution Architecture", "Enterprise Architecture", "Data Science", "Stakeholder Management",
    ]
    low = (text or "").lower()
    found: List[str] = []
    for s in lexicon:
        if re.search(r"\b" + re.escape(s.lower()) + r"\b", low):
            found.append(s)

    for line in (text or "").splitlines():
        m = re.match(r"^(?:skills?|technical skills|core competencies)\s*:\s*(.+)$", line.strip(), flags=re.I)
        if m:
            for part in re.split(r"[,;/|]", m.group(1)):
                token = part.strip(" -•·")
                if 2 <= len(token) <= 60:
                    found.append(token)
    return list(dict.fromkeys(found))[:120]


def _parse_experience(text: str) -> List[ExperienceModel]:
    out: List[ExperienceModel] = []
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    role_re = re.compile(r"(Engineer|Developer|Architect|Scientist|Manager|Lead|Consultant|Analyst)", re.I)
    for idx, line in enumerate(lines[:300]):
        if not role_re.search(line):
            continue
        nearby = " ".join(lines[idx: idx + 3])
        years = 0.0
        m = re.search(r"(20\d{2}|19\d{2})\s*[-–]\s*(Present|Current|Now|20\d{2}|19\d{2})", nearby, re.I)
        if m:
            start = int(m.group(1))
            end_raw = m.group(2).lower()
            end = 2026 if end_raw in {"present", "current", "now"} else int(m.group(2))
            years = max(0.0, float(end - start))
        bullets: List[str] = []
        for bline in lines[idx + 1: idx + 8]:
            if re.match(r"^[-•*]", bline):
                bullets.append(re.sub(r"^[-•*]\s*", "", bline))
        out.append(ExperienceModel(title=line, years=years, bullets=bullets[:10]))
        if len(out) >= 12:
            break
    return out


def _parse_education(text: str) -> List[EducationModel]:
    out: List[EducationModel] = []
    for line in (text or "").splitlines():
        clean = line.strip()
        if not clean:
            continue
        if re.search(r"\b(B\.?Tech|B\.?E|Bachelor|Master|M\.?S\.?|MBA|PhD|Doctorate)\b", clean, flags=re.I):
            yr = None
            m = re.search(r"(19\d{2}|20\d{2})", clean)
            if m:
                yr = m.group(1)
            out.append(EducationModel(institution=clean, graduation_year=yr))
    return out[:8]


def _parse_projects(text: str) -> List[str]:
    projects: List[str] = []
    for line in (text or "").splitlines():
        m = re.search(r"(?:project|projects)\s*[:\-]?\s*(.+)$", line, flags=re.I)
        if m:
            val = m.group(1).strip(" .-")
            if len(val) >= 6:
                projects.append(val)
    return list(dict.fromkeys(projects))[:12]


def _infer_project_skills_from_experience(experience: List[ExperienceModel]) -> List[str]:
    blob = "\n".join(b for exp in experience for b in (exp.bullets or []))
    if not blob:
        return []
    mappings = {
        "Agentic AI": ["agent", "multi-agent", "autonomous"],
        "LLM": ["llm", "gpt", "gemini", "claude"],
        "Machine Learning": ["model", "training", "inference", "predictive"],
        "MLOps": ["mlops", "deployment", "monitoring", "mlflow"],
        "Big Data": ["tb", "1tb", "spark", "pipeline", "large-scale"],
    }
    low = blob.lower()
    return [skill for skill, hints in mappings.items() if any(h in low for h in hints)]


def _infer_latent_skills(text: str, existing: List[str]) -> List[str]:
    blob = (text or "").lower()
    mapping = {
        "Solution Architecture": ["architecture", "solution design", "reference architecture"],
        "Data Science": ["forecast", "feature engineering", "ab test", "data science"],
        "Machine Learning": ["train", "inference", "ml pipeline"],
        "AI/ML": ["llm", "genai", "ai assistant", "rag"],
        "Cloud Engineering": ["aws", "azure", "gcp", "cloud"],
        "Stakeholder Management": ["stakeholder", "cross-functional", "business partner"],
    }
    seen = {s.lower() for s in (existing or [])}
    out: List[str] = []
    for skill, cues in mapping.items():
        if skill.lower() in seen:
            continue
        if any(cue in blob for cue in cues):
            out.append(skill)
    return out


class ParserEvaluatorL2:
    """Compares rapid/deep pass and decides merge with retry protection."""

    def __init__(self, retry_limit: int = 2, threshold: float = 0.85):
        self.retry_limit = retry_limit
        self.threshold = threshold

    def evaluate(self, phase1: ExtractedResume, phase2: ExtractedResume) -> Tuple[ExtractedResume, float, bool]:
        retries = 0
        best = phase1
        best_score = 0.0

        while retries <= self.retry_limit:
            p1_skills = {s.lower() for s in (phase1.skills or [])}
            p2_skills = {s.lower() for s in (phase2.skills or [])}
            skill_growth = (len(p2_skills) - len(p1_skills)) / max(1, len(p1_skills))
            missing_edu_in_p1 = len(phase1.education or []) == 0 and len(phase2.education or []) > 0

            merge_needed = skill_growth > 0.2 or missing_edu_in_p1
            merged = phase1.model_copy(deep=True)
            if merge_needed:
                merged.skills = list(dict.fromkeys((phase1.skills or []) + (phase2.skills or [])))
                merged.projects = list(dict.fromkeys((phase1.projects or []) + (phase2.projects or [])))
                if not merged.education and phase2.education:
                    merged.education = phase2.education
                if len(phase2.experience or []) > len(merged.experience or []):
                    merged.experience = phase2.experience

            score = 0.35
            score += 0.25 * min(1.0, len(merged.skills) / 20.0)
            score += 0.20 if merged.education else 0.0
            score += 0.20 if merged.projects else 0.0

            best, best_score = merged, score
            if score >= self.threshold:
                best.low_confidence = False
                best.evaluation_score = round(score, 3)
                return best, round(score, 3), False
            retries += 1

        best.low_confidence = True
        best.evaluation_score = round(best_score, 3)
        return best, round(best_score, 3), True


class ParserAgentService:
    """Dual-phase cognitive parser for L2 with evaluator loop."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.gemini = GeminiClient(settings)
        self.evaluator = ParserEvaluatorL2(retry_limit=2, threshold=0.85)

    @staticmethod
    def _coerce_llm_json(payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload.strip(), flags=re.I | re.S)
            try:
                out = json.loads(txt)
            except Exception:
                return None
            return out if isinstance(out, dict) else None
        return None

    def _phase2_deep_context(self, text: str, phase1: ExtractedResume) -> ExtractedResume:
        inferred = _infer_latent_skills(text, phase1.skills)
        inferred += _infer_project_skills_from_experience(phase1.experience)
        skills = list(dict.fromkeys((phase1.skills or []) + inferred))
        projects = _parse_projects(text)
        education = phase1.education or _parse_education(text)

        if self.s.GEMINI_API_KEY:
            prompt = (
                "Analyze achievements and notable projects. Infer implied skills. "
                "Return strict JSON: {skills:string[], projects:string[], education:string[]}.\n\n"
                f"RESUME_TEXT:\n{text[:14000]}"
            )
            j = self._coerce_llm_json(self.gemini.generate_json(prompt, temperature=0.0, max_tokens=900))
            if isinstance(j, dict):
                llm_skills = [str(s).strip() for s in (j.get("skills") or []) if str(s).strip()]
                llm_projects = [str(p).strip() for p in (j.get("projects") or []) if str(p).strip()]
                llm_edu = [str(e).strip() for e in (j.get("education") or []) if str(e).strip()]
                skills = list(dict.fromkeys(skills + llm_skills))
                projects = list(dict.fromkeys(projects + llm_projects))
                if not education and llm_edu:
                    education = [EducationModel(institution=e) for e in llm_edu[:6]]

        return ExtractedResume(
            name=phase1.name,
            contact=phase1.contact,
            summary=phase1.summary,
            skills=skills,
            experience=phase1.experience,
            education=education,
            projects=projects,
            total_years_experience=phase1.total_years_experience,
        )

    def parse_from_upload(self, *, filename: str, file_bytes: bytes, raw_text: Optional[str] = None) -> Tuple[ExtractedResume, str]:
        text = raw_text or ""
        if not text:
            if filename.lower().endswith(".docx"):
                text = _extract_docx_text(file_bytes)
            else:
                text = file_bytes.decode("utf-8", errors="ignore")

        experience = _parse_experience(text)
        total_years = sum(float(e.years or 0.0) for e in experience)

        phase1 = ExtractedResume(
            name=_guess_name(text),
            contact=ContactModel(email=_rx_email(text), phone=_rx_phone(text)),
            summary=" ".join([x.strip() for x in text.splitlines()[1:4] if x.strip()])[:500],
            skills=_parse_skills_phase1(text),
            experience=experience,
            education=_parse_education(text),
            projects=_parse_projects(text),
            total_years_experience=total_years,
        )

        phase2 = self._phase2_deep_context(text, phase1)
        merged, score, low_conf = self.evaluator.evaluate(phase1, phase2)
        merged.evaluation_score = score
        merged.low_confidence = low_conf
        return merged, text


__all__ = [
    "ParserAgentService",
    "ExtractedResume",
    "ExperienceModel",
    "EducationModel",
    "_infer_latent_skills",
    "_infer_project_skills_from_experience",
]
