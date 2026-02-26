from __future__ import annotations

import json
import re
import zipfile
from difflib import SequenceMatcher
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


DEFAULT_MASTER_TECH_STACK = [
    "Python", "Java", "JavaScript", "TypeScript", "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Airflow", "Spark", "Hadoop", "Kafka",
    "REST", "GraphQL", "FastAPI", "Django", "Flask", "React", "Node.js", "Pandas", "NumPy", "PyTorch",
    "TensorFlow", "Scikit-learn", "LLM", "Prompt Engineering", "CI/CD", "Git", "Linux", "Tableau", "Power BI",
]


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


def _load_master_tech_stack() -> List[str]:
    candidates = [
        Path("master_tech_stack.json"),
        Path("src/careeragent/artifacts/master_tech_stack.json"),
        Path(__file__).resolve().parents[3] / "master_tech_stack.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            raw = payload.get("skills") or payload.get("tech_stack") or []
        else:
            raw = payload
        if not isinstance(raw, list):
            continue
        cleaned = [_clean_skill_token(str(x) or "") for x in raw]
        items = [x for x in cleaned if x]
        if items:
            return items
    return DEFAULT_MASTER_TECH_STACK


def _fuzzy_match_token(token: str, master: List[str], threshold: float = 0.84) -> Optional[str]:
    t = _clean_skill_token(token)
    if not t:
        return None
    low = t.lower()
    for m in master:
        if low == m.lower():
            return m
    best: Optional[str] = None
    score = 0.0
    for m in master:
        s = SequenceMatcher(None, low, m.lower()).ratio()
        if s > score:
            score = s
            best = m
    if best and score >= threshold:
        return best
    return t


def _regex_skill_fallback(text: str, master: List[str]) -> List[str]:
    found: List[str] = []
    for skill in master:
        pattern = r"(?<!\w)" + re.escape(skill).replace(r"\ ", r"[\s\-/]+") + r"(?!\w)"
        if re.search(pattern, text or "", flags=re.I):
            found.append(skill)
    return found


def _extract_responsibility_signals(text: str) -> List[str]:
    """Promote seniority and architecture responsibilities into skill-like tags."""
    patterns = {
        "Principal Leadership": r"\bprincipal\b",
        "Architecture Strategy": r"\b(architect|architecture|solution architect|enterprise architect)\b",
        "Technical Leadership": r"\b(tech lead|technical lead|leadership|mentorship|mentoring)\b",
        "System Design": r"\b(system design|distributed systems|reference architecture)\b",
        "Stakeholder Management": r"\b(stakeholder|executive|cross-functional|roadmap)\b",
    }
    found: List[str] = []
    low = text or ""
    for label, pattern in patterns.items():
        if re.search(pattern, low, flags=re.I):
            found.append(label)
    return found


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
    master = _load_master_tech_stack()
    recovered = _extract_skill_like_lines(text)
    validated = SkillExtractionValidator.model_validate({"skills": list(current_skills or []) + recovered})
    deduped: List[str] = []
    seen = set()
    for skill in validated.skills:
        clean = _fuzzy_match_token(skill, master)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    if not deduped:
        for skill in _regex_skill_fallback(text, master):
            key = skill.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(skill)

    for signal in _extract_responsibility_signals(text):
        key = signal.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    return deduped[:100]


def _parse_experience(text: str) -> List[ExperienceModel]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    out: List[ExperienceModel] = []
    # very light heuristic: look for "Company" in next line
    for i, ln in enumerate(lines[:400]):
        if re.search(r"(Principal|Staff|Lead|Manager|Director|Solution Architect|Data Scientist|Engineer|Architect|Consultant)", ln, flags=re.I):
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

    @staticmethod
    def _coerce_llm_json(payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            txt = payload.strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I | re.S).strip()
            try:
                j = json.loads(txt)
            except Exception:
                return None
            return j if isinstance(j, dict) else None
        return None

    def _llm_skill_backfill(self, text: str, current_skills: List[str]) -> List[str]:
        if not self.s.GEMINI_API_KEY:
            return current_skills
        prompt = (
            "Extract a concise SKILLS list from the resume text. Include both technical and non-technical skills "
            "if explicitly present. Return STRICT JSON: {skills: string[]}. Do not invent data.\n\n"
            f"CURRENT_SKILLS: {current_skills}\n\nRESUME_TEXT:\n{text[:10000]}"
        )
        j = self._coerce_llm_json(self.gemini.generate_json(prompt, temperature=0.0, max_tokens=500))
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
            j = self._coerce_llm_json(self.gemini.generate_json(prompt, temperature=0.1, max_tokens=1000))
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
