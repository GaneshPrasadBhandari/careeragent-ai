from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated, TypedDict


def utc_now() -> str:
    """Description: Return UTC timestamp string. Layer: L0 Input: None Output: ISO string"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Description: Merge reducer for state dicts. Layer: L0 Input: dict + dict Output: merged dict"""
    out = dict(a or {})
    out.update(b or {})
    return out


@dataclass
class AttemptLog:
    """Description: Tool/model attempt log. Layer: L0 Input: tool run Output: log record"""
    attempt_id: str
    layer_id: str
    agent: str
    tool: str
    model: Optional[str]
    status: str  # ok | low_conf | failed
    confidence: float
    started_at_utc: str
    finished_at_utc: str
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class GateEvent:
    """Description: Recursive gate decision. Layer: L3-L7 Input: scores Output: pass/retry/hitl"""
    layer_id: str
    target: str
    score: float
    threshold: float
    decision: str  # pass | retry | hitl
    retries: int
    feedback: List[str]
    reasoning_chain: Optional[List[str]] = None
    at_utc: str = ""


class CareerGraphState(TypedDict, total=False):
    """
    Description: LangGraph state (Annotated reducers).
    Layer: L0-L9
    Input: nodes update slices
    Output: full orchestration state
    """

    run_id: str
    status: str  # running | needs_human_approval | blocked | completed | failed

    # config / overrides
    thresholds: Dict[str, float]  # e.g. {"parser":0.7,"discovery":0.7,"match":0.7,"draft":0.7}
    max_retries: int
    layer_retry_count: Dict[str, int]

    # user inputs
    preferences: Dict[str, Any]
    resume_bytes: Optional[bytes]
    resume_filename: Optional[str]
    resume_text: Optional[str]

    # extracted profile / intake bundle
    profile: Dict[str, Any]  # ExtractedResume json

    # jobs
    discovery_queries: List[str]
    jobs_raw: List[Dict[str, Any]]          # search results (title/link/snippet/source)
    jobs_scraped: List[Dict[str, Any]]      # with full_text, signals
    jobs_scored: List[Dict[str, Any]]       # with match components + score
    ranking: List[Dict[str, Any]]           # sorted shortlist

    # drafts
    drafts: Dict[str, Any]                  # tailored resume/cover + metadata
    bridge_docs: Dict[str, Any]             # missing skill -> links/doc path

    # HITL
    pending_action: Optional[str]
    hitl_reason: Optional[str]
    hitl_payload: Dict[str, Any]

    # audit
    live_feed: Annotated[List[Dict[str, Any]], operator.add]
    attempts: Annotated[List[AttemptLog], operator.add]
    gates: Annotated[List[GateEvent], operator.add]
    artifacts: Annotated[Dict[str, Any], merge_dicts]