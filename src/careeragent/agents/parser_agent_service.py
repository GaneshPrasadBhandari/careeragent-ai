from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from docx import Document
except Exception:  # pragma: no cover - optional dependency
    Document = None
from pydantic import BaseModel, Field

from careeragent.core.settings import Settings
from careeragent.tools.llm_tools import GeminiClient


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF using pypdf (correct package name)."""
    try:
        from pypdf import PdfReader  # pypdf >= 3.x
        reader = PdfReader(BytesIO(file_bytes))
        parts: List[str] = []
        for page in reader.pages[:20]:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t.strip())
        return "\n".join(parts)
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader as PdfReader2  # legacy fallback
        reader2 = PdfReader2(BytesIO(file_bytes))
        parts2: List[str] = []
        for page in reader2.pages[:20]:
            t = page.extract_text() or ""
            if t.strip():
                parts2.append(t.strip())
        return "\n".join(parts2)
    except Exception:
        pass
    return ""


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


def _parse_skills(text: str) -> List[str]:
    """Extract explicit and inferred skills from resume text."""
    t = text or ""
    skills: List[str] = []
    seen_lower: set[str] = set()

    def _add(s: str) -> None:
        k = re.sub(r"\s+", " ", s or "").strip().lower()
        if k and k not in seen_lower and len(k) > 1:
            seen_lower.add(k)
            skills.append(s.strip())

    # section-based extraction
    for line in t.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(?:core competencies|technical skills|key skills|skills?)\s*:\s*(.+)$", stripped, flags=re.I)
        if m:
            for part in re.split(r"[,;|/]", m.group(1)):
                token = re.sub(r"^[-•·]\s*", "", part).strip()
                if 1 < len(token) <= 60:
                    _add(token)

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


def _infer_project_skills_from_experience(experience: List[ExperienceModel]) -> List[str]:
    """Infer latent skills from project/experience bullets."""
    bullet_blob = "\n".join(
        b.strip() for exp in (experience or []) for b in (exp.bullets or []) if b and b.strip()
    )
    if not bullet_blob:
        return []
    mappings = {
        "Agentic AI": r"\b(agent|multi-agent|autonomous agent)\b",
        "LLM": r"\b(llm|gpt|gemini|claude|large language model)\b",
        "Machine Learning": r"\b(machine learning|ml model|predictive model|training pipeline)\b",
        "Orchestration": r"\b(orchestrat|workflow engine|langgraph|airflow|state machine)\b",
        "MLOps": r"\b(mlops|mlflow|model deployment|model monitoring)\b",
        "Enterprise Architecture": r"\b(enterprise architecture|reference architecture|solution architecture)\b",
    }
    inferred: List[str] = []
    for label, pattern in mappings.items():
        if re.search(pattern, bullet_blob, flags=re.I):
            inferred.append(label)
    return inferred


def _infer_latent_skills(text: str, existing: List[str]) -> List[str]:
    """Infer likely skills not explicitly listed but strongly implied by experience text."""
    blob = (text or "").lower()
    inferred: List[str] = []
    mapping = {
        "Solution Architecture": ["architecture", "solution design", "reference architecture"],
        "Data Science": ["forecast", "model", "feature engineering", "a/b test"],
        "Machine Learning": ["train", "inference", "ml pipeline"],
        "AI/ML": ["llm", "genai", "ai assistant", "rag"],
        "Cloud Engineering": ["aws", "azure", "gcp", "cloud"],
        "Stakeholder Management": ["stakeholder", "cross-functional", "business partner"],
    }
    seen = {s.lower() for s in existing if s}
    for skill, triggers in mapping.items():
        if skill.lower() in seen:
            continue
        if any(t in blob for t in triggers):
            inferred.append(skill)
    return inferred


def _extract_skill_like_lines(text: str) -> List[str]:
    out: List[str] = []
    capture = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1]
        parts = re.split(r"[,;|/·•\t]+", line)
        for p in parts:
            p = re.sub(r"^\s*[-•·]\s*", "", p.strip())
            p = re.sub(r"\s+", " ", p).strip()
            if p and 2 <= len(p) <= 60:
                _add(p)

    # --- Step 2: Vocabulary scan of full document ---
    VOCAB: List[str] = [
        # Languages
        "Python", "Java", "Scala", "Go", "Golang", "C++", "C#", ".NET", "R",
        "JavaScript", "TypeScript", "Bash", "Shell", "Perl", "Ruby", "Swift", "Kotlin",
        "SQL", "PL/SQL", "T-SQL", "HQL",
        # Cloud
        "AWS", "Azure", "GCP", "Google Cloud",
        "SageMaker", "Lambda", "EC2", "ECS", "EKS", "S3", "RDS", "DynamoDB",
        "Azure OpenAI", "Azure ML", "Azure DevOps", "Azure Blob",
        "Databricks", "Snowflake", "BigQuery", "Redshift",
        # DevOps / MLOps
        "Docker", "Kubernetes", "Helm", "Terraform", "Ansible",
        "GitHub Actions", "GitLab CI", "Jenkins", "CircleCI", "ArgoCD",
        "MLflow", "DVC", "Evidently", "Weights & Biases",
        "Airflow", "Prefect", "Dagster", "Kubeflow",
        "Kafka", "RabbitMQ", "Celery", "Redis",
        "Prometheus", "Grafana", "Datadog", "Splunk",
        # Databases
        "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Cassandra",
        "Elasticsearch", "Neo4j", "Pinecone", "Qdrant", "Chroma", "FAISS",
        # Data
        "Pandas", "NumPy", "Spark", "PySpark", "Dask", "Polars",
        "ETL", "ELT", "dbt", "Power BI", "Tableau", "Looker",
        # ML / AI
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "XGBoost", "LightGBM", "scikit-learn",
        "PyTorch", "TensorFlow", "Keras", "Hugging Face",
        # GenAI / Agents
        "GenAI", "LLM", "RAG", "LangChain", "LangGraph", "LangSmith",
        "OpenAI", "GPT-4", "Claude", "Gemini", "Llama", "Mistral", "Ollama",
        "CrewAI", "Vector Database", "Embeddings", "Prompt Engineering",
        "Function Calling", "Tool Calling", "MCP",
        "Fine-tuning", "RLHF", "PEFT", "LoRA",
        # APIs / Backend
        "FastAPI", "Flask", "Django", "Spring Boot",
        "REST API", "GraphQL", "gRPC", "OpenAPI", "Microservices",
        # Frontend
        "React", "Next.js", "Vue.js", "Streamlit", "Gradio",
        # Architecture
        "System Design", "Solution Architecture", "Enterprise Architecture",
        "Microservices", "Serverless", "Data Mesh", "Data Lakehouse",
    ]

    low = t.lower()
    for v in VOCAB:
        if re.search(r"\b" + re.escape(v.lower()) + r"\b", low) and v.lower() not in seen_lower:
            _add(v)

    return skills[:120]


def _parse_experience(text: str) -> List[ExperienceModel]:
    """Cognitive experience extraction: finds any role/company block regardless of title keyword."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    out: List[ExperienceModel] = []

    # Find the experience section
    exp_start = -1
    exp_end = len(lines)
    for i, ln in enumerate(lines):
        if re.search(r"^(Professional\s+)?Experience|Work\s+History|Employment", ln, flags=re.I):
            exp_start = i + 1
        if exp_start > 0 and i > exp_start and re.search(
            r"^(Education|Certifications?|Projects?|Skills|Awards|Publications|References|Professional\s+Affil)", ln, flags=re.I
        ):
            exp_end = i
            break

    if exp_start < 0:
        # No header found — scan all lines
        exp_start = 0

    # Date pattern for detecting role header lines
    DATE_PAT = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|"
        r"July|August|September|October|November|December)[\s\.,]*"
        r"(19|20)\d{2}"
        r"|Present|Current|(19|20)\d{2}\s*[-–—]\s*(Present|Current|(19|20)\d{2})",
        flags=re.I,
    )

    # Heuristic: a role header is a SHORT line (≤ 12 words) that has a date nearby
    current: Optional[Dict[str, Any]] = None
    for i in range(exp_start, min(exp_end, len(lines))):
        ln = lines[i]
        if not ln:
            continue

        words = ln.split()
        # Check if this line or next 2 lines has a date → this is a new role
        nearby = " ".join(lines[i : min(i + 3, len(lines))])
        has_date = bool(DATE_PAT.search(nearby))

        # Strong title signal: short non-bullet line with title case + date nearby
        is_title_line = (
            has_date
            and 1 <= len(words) <= 14
            and not ln.startswith(("-", "•", "·", "*", "–"))
            and not re.match(r"^\d", ln)
        )

        if is_title_line and not re.search(
            r"(phone|email|linkedin|github|http|www|@)", ln, flags=re.I
        ):
            if current:
                out.append(
                    ExperienceModel(
                        title=current["title"],
                        company=current.get("company"),
                        start_date=current.get("start_date"),
                        end_date=current.get("end_date"),
                        bullets=current["bullets"][:12],
                    )
                )
            # Extract dates from the line/nearby
            dates = DATE_PAT.findall(nearby)
            start_d = end_d = None
            dm = re.search(
                r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*"
                r"(19|20)\d{2})\s*[-–—]\s*"
                r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*"
                r"(19|20)\d{2}|Present|Current)",
                nearby, flags=re.I,
            )
            if dm:
                parts_d = re.split(r"\s*[-–—]\s*", dm.group(0), maxsplit=1)
                start_d = parts_d[0].strip() if parts_d else None
                end_d = parts_d[1].strip() if len(parts_d) > 1 else None

            # Company is usually the next non-bullet, non-date line
            company = None
            for j in range(i + 1, min(i + 4, len(lines))):
                nl = lines[j].strip()
                if not nl:
                    continue
                if DATE_PAT.search(nl) and len(nl.split()) < 8:
                    continue  # skip pure date lines
                if nl.startswith(("-", "•", "·", "*")):
                    break  # already into bullets
                company = nl
                break

            current = {
                "title": ln,
                "company": company,
                "start_date": start_d,
                "end_date": end_d,
                "bullets": [],
            }
            continue

        if current and re.match(r"^[-•·*–]\s*", ln):
            current["bullets"].append(re.sub(r"^[-•·*–]\s*", "", ln).strip())

    if current:
        out.append(
            ExperienceModel(
                title=current["title"],
                company=current.get("company"),
                start_date=current.get("start_date"),
                end_date=current.get("end_date"),
                bullets=current["bullets"][:12],
            )
        )

    # Fallback: if still empty, try the old title-keyword heuristic
    if not out:
        for i, ln in enumerate(lines[:400]):
            if re.search(
                r"(Solution\s+Architect|Data\s+Scientist|Engineer|Developer|Manager|"
                r"Analyst|Consultant|Director|Lead|Specialist|Researcher|Intern|"
                r"Associate|Senior|Junior|Staff|Principal)",
                ln, flags=re.I,
            ):
                company = None
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if nxt and not re.search(r"(phone|email|linkedin|github)", nxt, flags=re.I):
                        company = nxt
                bullets: List[str] = []
                for j in range(i + 1, min(i + 22, len(lines))):
                    if re.match(r"^[-•·*]\s*", lines[j]):
                        bullets.append(re.sub(r"^[-•·*]\s*", "", lines[j]).strip())
                out.append(ExperienceModel(title=ln, company=company, bullets=bullets[:10]))
        out = out[:12]

    return out[:15]


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
    if Document is None:
        return ""
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
            "and infer skills from experience bullets when evidence is present "
            "(example: 'led development of agents' => 'Agentic AI', 'Orchestration'). "
            "if explicitly present. Return STRICT JSON: {skills: string[]}. Do not invent data.\n\n"
            "Few-shot quality labeling examples:\n"
            "Example A: 'Principal AI Architect' -> skills include ['Principal Leadership', 'Architecture Strategy'] and quality_signal='10/10'.\n"
            "Example B: 'Solutions Architecture' -> skills include ['Solutions Architecture', 'System Design'] and quality_signal='10/10'.\n"
            "Only output JSON with key 'skills'.\n\n"
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
        elif filename.lower().endswith(".pdf"):
            # CRITICAL FIX: use pypdf, NOT raw bytes decode (which gives binary garbage)
            if not text:
                text = _extract_pdf_text(file_bytes)
        else:
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
        inferred_skills = _infer_project_skills_from_experience(det.experience)
        latent_skills = _infer_latent_skills(text, det.skills)
        if inferred_skills or latent_skills:
            det.skills = list(dict.fromkeys((det.skills or []) + inferred_skills + latent_skills))
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
                "summary, skills(list of canonical display names), "
                "experience(list of {title,company,start_date,end_date,bullets(list of strings)}), "
                "education(list of {degree,institution,graduation_year}). "
                "Do NOT invent data; use null if unknown. Include ALL experience entries with ALL bullet points.\n\n"
                "RESUME_TEXT:\n" + text[:16000]
            )
            j = self.gemini.generate_json(prompt, temperature=0.1, max_tokens=3000)
            if isinstance(j, dict):
                try:
                    llm = ExtractedResume.model_validate(j)
                    det = _merge(det, llm)
                except Exception:
                    pass

        return det, text



def _validate_and_backfill_skills(text: str, current_skills: List[str]) -> List[str]:
    """Merge explicit skills from profile with skills inferred from resume skill sections."""
    skills: List[str] = []
    seen = set()

    def _add_many(items: List[str]) -> None:
        for raw in items or []:
            item = re.sub(r"\s+", " ", str(raw or "").strip())
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            skills.append(item)

    _add_many([s for s in (current_skills or []) if s])

    section_hits: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip().strip("•-* ")
        if not stripped:
            continue
        if re.match(r"^(core competencies|skills?)\s*:\s*", stripped, flags=re.I):
            _, rhs = stripped.split(":", 1)
            section_hits.extend([p.strip() for p in re.split(r"[,|/]", rhs) if p.strip()])

    _add_many(section_hits)
    return skills
