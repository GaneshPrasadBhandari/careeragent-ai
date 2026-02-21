from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ApplicationSubmission(BaseModel):
    """
    Description: L7 submission artifact representing a completed application submit action.
    Layer: L7
    Input: Final resume + cover letter references and job metadata
    Output: Submission record with submission_id and timestamps
    """

    model_config = ConfigDict(extra="forbid")

    submission_id: str
    job_id: str
    channel: Literal["simulated"] = "simulated"

    resume_artifact_key: str
    cover_letter_artifact_key: str

    submitted_at_utc: str
    notes: Optional[str] = None
