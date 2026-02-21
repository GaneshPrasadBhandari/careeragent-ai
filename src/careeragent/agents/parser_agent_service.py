
from __future__ import annotations

import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class ContactInfo(BaseModel):
    """
    Description: Contact info extracted from resume text.
    Layer: L2
    Input: raw resume text
    Output: structured contact object
    """
    model_config = ConfigDict(extra="ignore")
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None


class ExperienceItem(BaseModel):
    """
    Description: Experience item extracted from resume text.
    Layer: L2
    Input: raw resume text
    Output: structured experience item
    """
    model_config = ConfigDict(extra="ignore")
    title: Optional[str] = None
    company: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    """
    Description: Education item extracted from resume text.
    Layer: L2
    Input: raw resume text
    Output: structured education item
    """
    model_config = ConfigDict(extra="ignore")
    degree: Optional[str] = None
    school: Optional[str] = None
    year: Optional[str] = None


COMMON_SKILLS = [
    "python","sql","fastapi","docker","kubernetes","mlflow","dvc",
    "aws","azure","gcp","sagemaker","azure ml","databricks","spark",
    "pytorch","tensorflow","scikit-learn","pandas","numpy",
    "langchain","langgraph","rag","vector","faiss","chroma",
    "terraform","github actions","ci/cd","kafka","airflow","snowflake"
]


def _find_email(text: str) -> Optional[str]:
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text or "")
    return m.group(0) if m else None


def _find_phone(text: str) -> Optional[str]:
    # handles +1 (xxx) xxx-xxxx, xxx-xxx-xxxx, etc
    m = re.search(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)\d{3}[\s\-]?\d{4}", text or "")
    return m.group(0).strip() if m else None


def _find_url(text: str, keyword: str) -> Optional[str]:
    # capture URLs containing linkedin/github
    pat = rf"(https?://[^\s]+{keyword}[^\s]*)"
    m = re.search(pat, text or "", flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def _split_sections(text: str) -> dict[str, str]:
    """
    Description: Split text into sections using heading keywords.
    Layer: L2
    Input: raw text
    Output: dict section->content
    """
    t = text or ""
    # Normalize headings
    headings = ["summary", "skills", "experience", "education", "projects", "certifications"]
    idxs = []
    low = t.lower()
    for h in headings:
        m = re.search(rf"(^|\n)\s*{h}\s*[:\-]?\s*", low)
        if m:
            idxs.append((m.start(), h))
    if not idxs:
        return {"full": t}

    idxs.sort(key=lambda x: x[0])
    out: dict[str, str] = {}
    for i, (pos, h) in enumerate(idxs):
        end = idxs[i + 1][0] if i + 1 < len(idxs) else len(t)
        out[h] = t[pos:end].strip()
    out["full"] = t
    return out


def _extract_skills(text: str) -> List[str]:
    low = (text or "").lower()
    found = []
    # 1) explicit skills section comma split
    m = re.search(r"skills\s*[:\-]\s*(.+)", text or "", flags=re.IGNORECASE)
    if m:
        chunk = m.group(1)
        parts = re.split(r"[,\|/•\n]+", chunk)
        found.extend([p.strip().lower() for p in parts if p.strip()])
    # 2) dictionary scan
    for s in COMMON_SKILLS:
        if s in low:
            found.append(s)
    # cleanup
    found = [re.sub(r"\s+", " ", x) for x in found]
    return sorted(list(dict.fromkeys([x for x in found if 2 <= len(x) <= 40])))[:50]


def _extract_name(text: str) -> Optional[str]:
    # first non-empty line that isn't a heading
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    cand = lines[0]
    if cand.lower().startswith(("summary", "skills", "experience", "education")):
        return None
    # avoid emails/phones
    if "@" in cand or re.search(r"\d{3}[\s\-]?\d{3}", cand):
        return None
    return cand[:60]


class ExtractedResume(BaseModel):
    """
    Description: Core parsed resume model for downstream layers (L3-L9).
    Layer: L2
    Input: raw resume text
    Output: structured resume JSON
    """
    model_config = ConfigDict(extra="ignore")
    name: Optional[str] = None
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    raw_text: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ParserAgentService:
    """
    Description: Convert raw resume string into ExtractedResume JSON (deterministic, ATS-friendly extraction).
    Layer: L2
    Input: raw resume string
    Output: ExtractedResume
    """
    def parse(self, *, raw_text: str, orchestration_state: Any, feedback: List[str] | None = None) -> ExtractedResume:
        text = raw_text or ""
        sections = _split_sections(text)

        name = _extract_name(text)
        email = _find_email(text)
        phone = _find_phone(text)
        linkedin = _find_url(text, "linkedin")
        github = _find_url(text, "github")

        skills = _extract_skills(sections.get("skills") or text)

        summary = None
        if "summary" in sections:
            summary = re.sub(r"(?i)^summary\s*[:\-]?\s*", "", sections["summary"]).strip()
        else:
            # fallback: first 2-3 lines after name
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) >= 2:
                summary = " ".join(lines[1:3])[:400]

        # Experience: keep it simple but structured
        exp_items: List[ExperienceItem] = []
        exp_text = sections.get("experience")
        if exp_text:
            body = re.sub(r"(?i)^experience\s*[:\-]?\s*", "", exp_text).strip()
            bullets = [b.strip("-• ").strip() for b in re.split(r"\n+", body) if b.strip()]
            bullets = [b for b in bullets if len(b) > 15][:10]
            if bullets:
                exp_items.append(ExperienceItem(bullets=bullets))

        edu_items: List[EducationItem] = []
        edu_text = sections.get("education")
        if edu_text:
            body = re.sub(r"(?i)^education\s*[:\-]?\s*", "", edu_text).strip()
            # naive degree/school capture
            edu_items.append(EducationItem(degree=body[:120]))

        warnings: List[str] = []
        if not email and not phone:
            warnings.append("Missing email/phone in resume text.")
        if len(skills) < 5:
            warnings.append("Low skill extraction count; consider adding a clearer Skills section.")
        if not exp_items:
            warnings.append("Experience section not clearly detected.")

        return ExtractedResume(
            name=name,
            contact=ContactInfo(email=email, phone=phone, linkedin=linkedin, github=github),
            summary=summary,
            skills=skills,
            experience=exp_items,
            education=edu_items,
            raw_text=text,
            warnings=warnings,
        )
