from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
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
    medium: Optional[str] = None


class ExperienceModel(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class EducationModel(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    graduation_year: Optional[str] = None


class ExtractedResume(BaseModel):
    """Description: Canonical extracted resume bundle.
    Layer: L2
    Input: resume text + doc links
    Output: structured profile
    """

    name: Optional[str] = None
    contact: ContactModel = Field(default_factory=ContactModel)
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceModel] = Field(default_factory=list)
    education: List[EducationModel] = Field(default_factory=list)


class SkillExtractionValidator(BaseModel):
    """Structured validator to enforce deterministic skill recovery before L3."""

    skills: List[str] = Field(default_factory=list)


def _rx_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    return m.group(0) if m else None


def _rx_phone(text: str) -> Optional[str]:
    m = re.search(r"(\+?\d{1,3}[-\s]?)?(\(?\d{3}\)?[-\s]?)\d{3}[-\s]?\d{4}", text or "")
    if not m:
        return None
    p = m.group(0).strip()
    return p if len(re.sub(r"\D", "", p)) >= 10 else None


def _rx_location(text: str) -> Optional[str]:
    head = "\n".join((text or "").splitlines()[:14])
    m = re.search(r"\b([A-Za-z][A-Za-z .'-]{2,}),\s*([A-Z]{2})\b", head)
    if m:
        return f"{m.group(1).strip()}, {m.group(2).strip()}"
    if re.search(r"\b(United States|USA|U\.S\.)\b", head, flags=re.I):
        return "United States"
    return None


def _guess_name(text: str) -> Optional[str]:
    for line in (text or "").splitlines()[:10]:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in ["phone", "email", "linkedin", "github", "http", "www"]):
            continue
        if len(s.split()) <= 6 and all(ch.isalpha() or ch in " .-'" for ch in s):
            return s
    return None


def _clean_skill_token(token: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", (token or "").strip(" -•\t"))
    if len(s) < 2 or len(s) > 48:
        return None
    if re.search(r"[^A-Za-z0-9 .+#&()/-]", s):
        return None
    if s.lower() in {"skills", "experience", "education", "summary", "projects"}:
        return None
    return s


def _extract_skill_like_lines(text: str) -> List[str]:
    out: List[str] = []
    capture = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            if capture:
                break
            continue
        low = line.lower()
        if re.search(r"\b(key skills|skills|core competencies|competencies|tools|technologies|expertise)\b", low):
            capture = True
            payload = line.split(":", 1)[1] if ":" in line else ""
            if payload:
                out.extend(re.split(r"[,;/|]\s*", payload))
            continue
        if capture:
            if re.search(r"\b(experience|education|projects|certifications|summary)\b", low):
                break
            out.extend(re.split(r"[,;/|]\s*", line))
    return out


def _parse_skills(text: str) -> List[str]:
    skills: List[str] = []
    for token in _extract_skill_like_lines(text):
        clean = _clean_skill_token(token)
        if clean and clean.lower() not in {x.lower() for x in skills}:
            skills.append(clean)
    return skills[:80]


def _validate_and_backfill_skills(text: str, current_skills: List[str]) -> List[str]:
    """Recover missing skills without hard-coded domain assumptions."""
    recovered = _extract_skill_like_lines(text)
    validated = SkillExtractionValidator.model_validate({"skills": list(current_skills or []) + recovered})
    deduped: List[str] = []
    seen = set()
    for skill in validated.skills:
        clean = _clean_skill_token(skill)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped[:100]


def _parse_experience(text: str) -> List[ExperienceModel]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    out: List[ExperienceModel] = []
    # very light heuristic: look for "Company" in next line
    for i, ln in enumerate(lines[:400]):
        if re.search(r"(Solution Architect|Data Scientist|Engineer|Architect|Consultant)", ln, flags=re.I):
            title = ln
            company = None
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not re.search(r"(phone|email|linkedin|github)", nxt, flags=re.I):
                    company = nxt
            bullets: List[str] = []
            for j in range(i + 1, min(i + 22, len(lines))):
                if lines[j].startswith("-") or lines[j].startswith("•"):
                    bullets.append(lines[j].lstrip("-• ").strip())
            out.append(ExperienceModel(title=title, company=company, bullets=bullets[:10]))
    return out[:12]


def _parse_education(text: str) -> List[EducationModel]:
    t = text or ""
    out: List[EducationModel] = []
    m = re.search(r"Education\s*(.*?)(Certifications|Projects|Professional Affiliations|$)", t, flags=re.S | re.I)
    if not m:
        return out
    block = m.group(1)
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        yr = None
        my = re.search(r"(19\d{2}|20\d{2})", line)
        if my:
            yr = my.group(1)
        out.append(EducationModel(institution=line, graduation_year=yr))
    return out[:8]


def _extract_docx_visible_text(docx_bytes: bytes) -> str:
    doc = Document(BytesIO(docx_bytes))
    parts = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    return "\n".join(parts)


def _extract_docx_hyperlinks(docx_bytes: bytes) -> List[str]:
    """Extract hyperlink targets from DOCX relationship XML.

    Word often shows "LinkedIn | GitHub" as visible text, but the URL only exists
    in word/_rels/document.xml.rels.
    """
    links: List[str] = []
    try:
        z = zipfile.ZipFile(BytesIO(docx_bytes))
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8", errors="ignore")
        # Relationship Target="https://..."
        for m in re.finditer(r"Target=\"(https?://[^\"]+)\"", rels, flags=re.I):
            url = m.group(1).strip().rstrip("/")
            if url and url not in links:
                links.append(url)
    except Exception:
        return links
    return links


def _merge(primary: ExtractedResume, backfill: ExtractedResume) -> ExtractedResume:
    merged = primary.model_copy(deep=True)

    if not merged.name and backfill.name:
        merged.name = backfill.name
    if not merged.summary and backfill.summary:
        merged.summary = backfill.summary

    # contact
    if not merged.contact.email and backfill.contact.email:
        merged.contact.email = backfill.contact.email
    if not merged.contact.phone and backfill.contact.phone:
        merged.contact.phone = backfill.contact.phone
    if not merged.contact.location and backfill.contact.location:
        merged.contact.location = backfill.contact.location

    # union links
    all_links = list(dict.fromkeys((merged.contact.links or []) + (backfill.contact.links or [])))
    merged.contact.links = all_links

    for u in all_links:
        if not merged.contact.linkedin and "linkedin.com" in u:
            merged.contact.linkedin = u
        if not merged.contact.github and "github.com" in u:
            merged.contact.github = u
        if not merged.contact.medium and "medium.com" in u:
            merged.contact.medium = u

    # skills union
    skills: List[str] = []
    seen = set()
    for s in (merged.skills or []) + (backfill.skills or []):
        if not s:
            continue
        k = s.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        skills.append(s.strip())
    merged.skills = skills[:100]

    # experience/education backfill only if empty
    if not merged.experience and backfill.experience:
        merged.experience = backfill.experience[:12]
    if not merged.education and backfill.education:
        merged.education = backfill.education[:10]

    return merged


class ParserAgentService:
    """Description: Dual parser (deterministic + Gemini backfill) with DOCX hyperlink extraction.
    Layer: L2
    Input: resume bytes/text
    Output: ExtractedResume
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.gemini = GeminiClient(settings)

    def _llm_skill_backfill(self, text: str, current_skills: List[str]) -> List[str]:
        if not self.s.GEMINI_API_KEY:
            return current_skills
        prompt = (
            "Extract a concise SKILLS list from the resume text. Include both technical and non-technical skills "
            "if explicitly present. Return STRICT JSON: {skills: string[]}. Do not invent data.\n\n"
            f"CURRENT_SKILLS: {current_skills}\n\nRESUME_TEXT:\n{text[:10000]}"
        )
        j = self.gemini.generate_json(prompt, temperature=0.0, max_tokens=500)
        if not isinstance(j, dict):
            return current_skills
        try:
            parsed = SkillExtractionValidator.model_validate(j)
        except Exception:
            return current_skills
        return _validate_and_backfill_skills(text, list(current_skills or []) + list(parsed.skills or []))

    def parse_from_upload(self, *, filename: str, file_bytes: bytes, raw_text: Optional[str] = None) -> Tuple[ExtractedResume, str]:
        """Return extracted profile + raw_text used."""
        text = raw_text or ""
        links: List[str] = []

        if filename.lower().endswith(".docx"):
            text = _extract_docx_visible_text(file_bytes)
            links = _extract_docx_hyperlinks(file_bytes)
        elif filename.lower().endswith(".txt"):
            text = file_bytes.decode("utf-8", errors="ignore")
        else:
            # PDF or unknown: assume raw_text already extracted elsewhere
            if not text:
                text = file_bytes.decode("utf-8", errors="ignore")

        name = _guess_name(text)
        email = _rx_email(text)
        phone = _rx_phone(text)
        location = _rx_location(text)

        det_links = list(dict.fromkeys(links))
        # also scrape visible URLs if present
        for u in re.findall(r"https?://\S+", text):
            u = u.strip().rstrip(").,]")
            if u not in det_links:
                det_links.append(u)

        contact = ContactModel(email=email, phone=phone, location=location, links=det_links)
        for u in det_links:
            if not contact.linkedin and "linkedin.com" in u:
                contact.linkedin = u
            if not contact.github and "github.com" in u:
                contact.github = u
            if not contact.medium and "medium.com" in u:
                contact.medium = u

        # summary heuristic
        summary = None
        m = re.search(r"Professional Summary\s*(.*?)(Key Skills|Skills|Professional Experience|Experience)\b", text, flags=re.S | re.I)
        if m:
            summary = re.sub(r"\s+", " ", m.group(1)).strip()[:1000]

        det = ExtractedResume(
            name=name,
            contact=contact,
            summary=summary,
            skills=_parse_skills(text),
            experience=_parse_experience(text),
            education=_parse_education(text),
        )
        det.skills = _validate_and_backfill_skills(text, det.skills)
        if len(det.skills) < 8:
            det.skills = self._llm_skill_backfill(text, det.skills)

        # LLM backfill if missing critical fields or empty experience
        missing_critical = (
            not det.name
            or not det.contact.email
            or not det.contact.phone
            or len(det.skills) < 10
            or len(det.experience) == 0
        )

        if missing_critical and self.s.GEMINI_API_KEY:
            prompt = (
                "Extract resume into STRICT JSON with keys: name, contact{email,phone,location,links,linkedin,github,medium}, "
                "summary, skills(list), experience(list of {title,company,start_date,end_date,bullets}), education(list). "
                "Do NOT invent data; use null if unknown.\n\nRESUME_TEXT:\n" + text[:12000]
            )
            j = self.gemini.generate_json(prompt, temperature=0.1, max_tokens=1000)
            if isinstance(j, dict):
                try:
                    llm = ExtractedResume.model_validate(j)
                    det = _merge(det, llm)
                    det.skills = _validate_and_backfill_skills(text, det.skills)
                    if len(det.skills) < 8:
                        det.skills = self._llm_skill_backfill(text, det.skills)
                except Exception:
                    pass

        return det, text
