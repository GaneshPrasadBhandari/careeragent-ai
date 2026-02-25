from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Set


# ---
# Deterministic skill/entity extraction.
#
# Why this exists:
# - Token-frequency ATS scoring is noisy and can collapse to tiny values (0.05) depending on tokenization.
# - JD alignment should be entity/skill-based and stable.
# - We keep this dependency-free so it runs everywhere (local, CI, classroom demos).
# ---


_STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "with",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "these",
    "those",
    "you",
    "we",
    "our",
    "your",
    "their",
    "role",
    "years",
    "year",
    "experience",
    "responsibilities",
    "requirements",
}


_SKILL_LEXICON = {
    # Languages
    "python",
    "java",
    "scala",
    "go",
    "golang",
    "javascript",
    "typescript",
    "sql",
    "bash",
    # Cloud
    "aws",
    "azure",
    "gcp",
    "sagemaker",
    "lambda",
    "ec2",
    "ecr",
    "eks",
    "s3",
    "azure openai",
    "azure ml",
    "databricks",
    # DevOps/MLOps
    "docker",
    "kubernetes",
    "helm",
    "terraform",
    "github actions",
    "cicd",
    "mlflow",
    "dvc",
    "airflow",
    "kafka",
    "prometheus",
    "grafana",
    "evidently",
    # Data
    "pandas",
    "numpy",
    "spark",
    "pyspark",
    "etl",
    "elt",
    # ML/AI
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "xgboost",
    "lightgbm",
    "sklearn",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "huggingface",
    # GenAI/Agents
    "genai",
    "llm",
    "rag",
    "langchain",
    "langgraph",
    "vector database",
    "qdrant",
    "chroma",
    "faiss",
    "pinecone",
    "mcp",
    "langsmith",
    "prompt engineering",
    "tool calling",
    "function calling",
    # Product / architecture
    "system design",
    "solution architecture",
    "enterprise architecture",
    "microservices",
    "fastapi",
    "rest api",
}


_MULTIWORD = sorted([s for s in _SKILL_LEXICON if " " in s], key=len, reverse=True)
_SINGLEWORD = {s for s in _SKILL_LEXICON if " " not in s}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def normalize_skill(skill: str) -> str:
    """Normalize skill string to a canonical form."""
    s = _norm(skill)
    aliases = {
        "ml": "machine learning",
        "ai": "machine learning",
        "gen ai": "genai",
        "azureopenai": "azure openai",
        "scikit learn": "scikit-learn",
        "k8s": "kubernetes",
        "gha": "github actions",
        "ci/cd": "cicd",
    }
    return aliases.get(s, s)


def extract_skills(text: str, *, extra_candidates: Iterable[str] | None = None) -> List[str]:
    """Extract skills/entities from free text.

    Description: Deterministic skill extraction used for JD alignment.
    Layer: L4
    Input: raw text
    Output: deduped, sorted skills
    """

    t = _norm(text)
    found: Set[str] = set()

    # multiword first
    for phrase in _MULTIWORD:
        if phrase in t:
            found.add(phrase)

    # single tokens
    toks = [x for x in re.split(r"[^a-z0-9+.#]", t) if x]
    for tok in toks:
        if tok in _STOPWORDS:
            continue
        if tok in _SINGLEWORD:
            found.add(tok)

    # acronyms
    acr = set(re.findall(r"\b[A-Z]{2,6}\b", text or ""))
    for a in acr:
        low = a.lower()
        if low in {"rag", "llm", "nlp", "ml", "ai"}:
            found.add(normalize_skill(low))

    # candidate skills (profile.skills) if they appear
    if extra_candidates:
        for s in extra_candidates:
            ns = normalize_skill(s)
            if not ns or ns in _STOPWORDS:
                continue
            if " " in ns:
                if ns in t:
                    found.add(ns)
            else:
                if re.search(rf"\b{re.escape(ns)}\b", t):
                    found.add(ns)

    return sorted({normalize_skill(x) for x in found if x})


@dataclass
class AlignmentScorecard:
    jd_alignment_percent: float
    missing_skills_gap_percent: float
    matched_jd_skills: List[str]
    missing_jd_skills: List[str]


def compute_jd_alignment(*, jd_text: str, resume_skills: Iterable[str]) -> AlignmentScorecard:
    """Compute entity-based JD alignment between JD skills and resume skills.

    Description: Entity-based JD alignment (skills) instead of token frequency.
    Layer: L4/L5
    Input: JD text + resume skills
    Output: alignment scorecard with gap
    """

    rs = {normalize_skill(x) for x in resume_skills if x}
    jd = set(extract_skills(jd_text, extra_candidates=rs))

    if not jd:
        return AlignmentScorecard(0.0, 100.0, [], [])

    matched = sorted([s for s in jd if s in rs])
    missing = sorted([s for s in jd if s not in rs])

    align = len(matched) / max(1, len(jd))
    gap = 1.0 - align

    return AlignmentScorecard(
        jd_alignment_percent=round(align * 100.0, 2),
        missing_skills_gap_percent=round(gap * 100.0, 2),
        matched_jd_skills=matched,
        missing_jd_skills=missing,
    )
