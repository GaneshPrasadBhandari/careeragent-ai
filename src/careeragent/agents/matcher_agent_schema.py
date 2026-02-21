from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobDescription(BaseModel):
    """
    Description: Canonical job description artifact for matching.
    Layer: L4
    Input: Parsed job post JSON from ingestion
    Output: Normalized JobDescription for downstream agents
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    role_title: str
    company: str
    country_code: str = "US"

    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)

    requirements_text: str = ""
    applicants_count: Optional[int] = None
    market_competition_factor: Optional[float] = None  # if provided, must be >= 1.0

    meta: Dict[str, Any] = Field(default_factory=dict)


class MatchComponents(BaseModel):
    """
    Description: Deterministic component scores for matching.
    Layer: L4
    Input: Resume + JobDescription
    Output: Normalized component scores [0,1] + market factor
    """

    model_config = ConfigDict(extra="forbid")

    skill_overlap: float
    experience_alignment: float
    ats_score: float
    market_competition_factor: float


class MatchReport(BaseModel):
    """
    Description: Matching output between ExtractedResume and JobDescription.
    Layer: L4
    Input: ExtractedResume + JobDescription
    Output: MatchReport with skill gaps + InterviewChanceScore
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    role_title: str
    company: str

    matched_skills: List[str] = Field(default_factory=list)
    missing_required_skills: List[str] = Field(default_factory=list)
    missing_preferred_skills: List[str] = Field(default_factory=list)

    components: MatchComponents
    interview_chance_score: float  # [0,1]
    overall_match_percent: float   # [0,100]

    rationale: List[str] = Field(default_factory=list)
