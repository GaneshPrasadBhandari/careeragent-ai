from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionItem(BaseModel):
    """
    Description: Concrete action the user can take to improve match and interview odds.
    Layer: L5
    Input: MatchReport + user constraints
    Output: ActionItem list
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    why_it_matters: str
    how_to_execute: List[str] = Field(default_factory=list)
    priority: str = "medium"  # low/medium/high
    eta_days: Optional[int] = None


class PivotStrategy(BaseModel):
    """
    Description: Strategy artifact that reframes experience if match is low.
    Layer: L5
    Input: MatchReport
    Output: PivotStrategy (action plan)
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    overall_match_percent: float
    posture: str  # "proceed", "proceed_with_edits", "pivot"
    action_items: List[ActionItem] = Field(default_factory=list)
