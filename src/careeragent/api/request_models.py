
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class AnalyzeResponse(BaseModel):
    """
    Description: Response model for /analyze.
    Layer: L8
    Input: run_id + status
    Output: API response payload
    """
    model_config = ConfigDict(extra="ignore")
    run_id: str
    status: str


class ActionRequest(BaseModel):
    """
    Description: Request model for /action/{run_id}.
    Layer: L5
    Input: action_type + payload
    Output: normalized action request
    """
    model_config = ConfigDict(extra="ignore")

    action_type: str = Field(
        ...,
        description="execute_layer|approve_ranking|reject_ranking|approve_drafts|reject_drafts|approve_job|reject_job|resume_cleanup_submit|..."
    )
    payload: Dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """
    Description: Generic /status response wrapper (optional).
    Layer: L8
    Input: full state dict
    Output: typed wrapper (if used)
    """
    model_config = ConfigDict(extra="allow")
