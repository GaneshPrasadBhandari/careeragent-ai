
from __future__ import annotations

import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ContactInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


class ExperienceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bullets: List[str] = Field(default_factory=list)


class ExtractedResume(BaseModel):
    """
    Description: L2 Intake Bundle (profile extracted from raw resume).
    Layer: L2
    Input: raw resume text
    Output: structured profile JSON
    """
    model_config = ConfigDict(extra="ignore")
    name: Optional[str] = None
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    raw_text: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump()


COMMON_SKILLS = [
    "python","sql","fastapi","docker","kubernetes","mlflow","dvc","aws","azure","gcp",
    "pytorch","tensorflow","scikit-learn","pandas","numpy","spark","databricks","snowflake",
    "langchain","langgraph","rag","faiss","chroma","terraform","github actions","kafka","airflow"
]


def _find_email(text: str) -> Optional[str]:
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text or "")
    return m.group(0) if m else None


def _find_phone(text: str) -> Optional[str]:
    m = re.search(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)\d{3}[\s\-]?\d{4}", text or "")
    return m.group(0).strip() if m else None


def _find_url(text: str, kw: str) -> Optional[str]:
    m = re.search(rf"(https?://[^\s]*{kw}[^\s]*)", text or "", flags=re.I)
    return m.group(1) if m else None


def _extract_name(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    cand = lines[0]
    if "@" in cand:
        return None
    return cand[:60]


def _extract_skills(text: str) -> List[str]:
    low = (text or "").lower()
    found = []
    for s in COMMON_SKILLS:
        if s in low:
            found.append(s)
    # also parse a "skills:" line
    m = re.search(r"skills\s*[:\-]\s*(.+)", text or "", flags=re.I)
    if m:
        parts = re.split(r"[,\|/•\n]+", m.group(1))
        found.extend([p.strip().lower() for p in parts if p.strip()])
    # unique + clean
    out = []
    for x in found:
        x = re.sub(r"\s+", " ", x.strip())
        if 2 <= len(x) <= 40 and x not in out:
            out.append(x)
    return out[:60]


class ParserAgentService:
    """
    Description: Deterministic ParserAgent (CareerOS-style). Never blocks automation.
    Layer: L2
    Input: raw resume text
    Output: ExtractedResume
    """
    def parse(self, *, raw_text: str, orchestration_state: Any, feedback: List[str] | None = None) -> ExtractedResume:
        text = raw_text or ""
        email = _find_email(text)
        phone = _find_phone(text)
        linkedin = _find_url(text, "linkedin")
        github = _find_url(text, "github")

        skills = _extract_skills(text)

        # summary: first 2-3 non-empty lines after name
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        name = _extract_name(text)
        summary = None
        if len(lines) >= 3:
            summary = " ".join(lines[1:3])[:400]

        # experience bullets: take lines starting with dash/bullet
        bullets = []
        for ln in lines:
            if ln.startswith(("-", "•")) and len(ln) > 15:
                bullets.append(ln.strip("-• ").strip())
        exp = [ExperienceItem(bullets=bullets[:12])] if bullets else []

        warnings = []
        if not (email or phone):
            warnings.append("Contact info missing (email/phone).")
        if len(skills) < 6:
            warnings.append("Low extracted skills; add a clearer Skills section.")
        if not bullets:
            warnings.append("No bullet points detected; add bullets under Experience for ATS.")

        return ExtractedResume(
            name=name,
            contact=ContactInfo(email=email, phone=phone, linkedin=linkedin, github=github),
            summary=summary,
            skills=skills,
            experience=exp,
            raw_text=text,
            warnings=warnings,
        )
