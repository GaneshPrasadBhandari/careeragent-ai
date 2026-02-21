from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CoverLetterDraft(BaseModel):
    """
    Description: Country-specific cover letter draft artifact.
    Layer: L6
    Input: MatchReport + Resume + JobDescription
    Output: Draft text for export + approval gate
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    country_code: str
    role_title: str
    company: str

    contact_block_included: bool = False
    subject_line: Optional[str] = None
    body: str

    highlighted_skills: List[str] = Field(default_factory=list)
