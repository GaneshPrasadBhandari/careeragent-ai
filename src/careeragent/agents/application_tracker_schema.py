from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ApplicationStatus = Literal["applied", "interviewing", "rejected"]


class StatusUpdateEvent(BaseModel):
    """
    Description: L8 status event representing an application state change over time.
    Layer: L8
    Input: submission_id + status + note
    Output: Immutable status update event for audit and analytics
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    submission_id: str
    job_id: str
    status: ApplicationStatus
    occurred_at_utc: str
    note: Optional[str] = None
