from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Backward-compatible UTC datetime helper."""
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    """Backward-compatible ISO formatter."""
    return dt.isoformat()


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return _iso_utc(_utc_now())


LayerId = Literal["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]


class ArtifactRef(BaseModel):
    """Description: Reference to an artifact saved on disk.
    Layer: L8
    Input: local file path
    Output: metadata reference
    """

    path: str
    mime: str = "application/json"
    created_at_utc: str = Field(default_factory=utc_now_iso)


class StepTrace(BaseModel):
    """Description: Step execution trace for UI timeline.
    Layer: L2
    Input: node events
    Output: list of step traces
    """

    layer_id: LayerId
    status: Literal["queued", "running", "ok", "error", "blocked"] = "queued"
    agent: str = ""
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    message: Optional[str] = None


class LiveFeedEvent(BaseModel):
    """Description: Streamable event for UI live feed.
    Layer: L2
    Input: node log event
    Output: event dict
    """

    layer: str
    agent: str
    message: str
    ts_utc: str = Field(default_factory=utc_now_iso)


class EvaluationEntry(BaseModel):
    """Description: Structured evaluation entry rendered by dashboard.
    Layer: L2
    Input: evaluator decision
    Output: state.evaluations entry
    """

    layer_id: str
    target_id: str
    evaluation_score: float
    threshold: float
    decision: Literal["pass", "retry", "fail"]
    feedback: List[str] = Field(default_factory=list)
    ts_utc: str = Field(default_factory=utc_now_iso)


class Preferences(BaseModel):
    """Description: User constraints and targets.
    Layer: L1
    Input: UI preferences
    Output: normalized preferences
    """

    target_roles: List[str] = Field(default_factory=list)
    country: str = "US"
    location: str = "United States"
    remote: bool = True
    wfo_ok: bool = True
    salary: str = ""
    visa_sponsorship_required: bool = False

    recency_hours: float = 36.0
    max_jobs: int = 40

    discovery_threshold: float = 0.7
    max_refinements: int = 3
    resume_threshold: float = 0.40  # lowered from 0.55 — imperfect parse should still continue
    draft_count: int = 10

    user_phone: Optional[str] = None


class PlannerPersona(BaseModel):
    """Description: One search persona plan.
    Layer: L2
    Input: resume + goals
    Output: persona plan
    """

    persona_id: str
    name: str
    strategy: Literal["ats_only", "ats_preferred", "broad"] = "ats_preferred"
    recency_hours: float = 36.0
    must_include: List[str] = Field(default_factory=list)
    negative_terms: List[str] = Field(default_factory=list)
    site_filters: List[str] = Field(default_factory=list)


class BestSoFar(BaseModel):
    """Description: Best ranking preserved across retries.
    Layer: L2
    Input: ranking snapshots
    Output: best snapshot pointers
    """

    ranking_path: Optional[str] = None
    jobs_scored_path: Optional[str] = None
    score: float = 0.0
    persona_id: Optional[str] = None


class AgentState(BaseModel):
    """Description: Main orchestration state for CareerAgent-AI.
    Layer: L2
    Input: user inputs + intermediate outputs
    Output: full run state JSON
    """

    run_id: str

    status: Literal["queued", "running", "needs_human_approval", "completed", "failed", "blocked", "api_failure"] = "queued"
    pending_action: Optional[str] = None

    current_layer: Optional[str] = None
    progress_percent: int = 0
    updated_at_utc: str = Field(default_factory=utc_now_iso)

    preferences: Preferences = Field(default_factory=Preferences)

    # --- Outputs
    extracted_profile: Dict[str, Any] = Field(default_factory=dict)
    jobs_raw: List[Dict[str, Any]] = Field(default_factory=list)
    jobs_scored: List[Dict[str, Any]] = Field(default_factory=list)
    ranking: List[Dict[str, Any]] = Field(default_factory=list)

    approved_job_urls: List[str] = Field(default_factory=list)

    # --- L6 Draft output metadata (populated by DraftingAgentService)
    drafts: List[Dict[str, Any]] = Field(default_factory=list)

    # --- Planner/Director
    search_personas: List[PlannerPersona] = Field(default_factory=list)
    active_persona_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 6

    refinement_feedback: Optional[str] = None
    query_modifiers: Dict[str, Any] = Field(default_factory=dict)

    best_so_far: BestSoFar = Field(default_factory=BestSoFar)

    # --- Evaluations / logs
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    evaluations: List[EvaluationEntry] = Field(default_factory=list)
    evaluation_logs: List[str] = Field(default_factory=list)

    # --- UI trace
    steps: List[StepTrace] = Field(default_factory=list)
    live_feed: List[LiveFeedEvent] = Field(default_factory=list)

    # --- Artifacts
    artifacts: Dict[str, ArtifactRef] = Field(default_factory=dict)

    # --- Meta
    meta: Dict[str, Any] = Field(default_factory=dict)

    # --- Cost / governance
    token_budget: int = 120_000
    token_used_est: int = 0
    robots_violations: List[str] = Field(default_factory=list)

    def feed(self, layer: str, agent: str, message: str) -> None:
        """Description: Append a live feed event.
        Layer: L2
        Input: layer + message
        Output: state.live_feed
        """
        self.live_feed.append(LiveFeedEvent(layer=layer, agent=agent, message=message))

    def log_eval(self, msg: str) -> None:
        """Description: Append verbose evaluator log.
        Layer: L2
        Input: message
        Output: state.evaluation_logs
        """
        self.evaluation_logs.append(msg)


    def end_step_ok(self, layer_id: LayerId, message: str = "ok") -> None:
        """Description: Mark step ok.
        Layer: L2
        Input: layer id
        Output: steps update
        """
        for s in reversed(self.steps):
            if s.layer_id == layer_id and s.status == "running":
                s.status = "ok"
                s.finished_at_utc = utc_now_iso()
                s.message = message
                break
        self.feed(layer_id, "orchestrator", f"{layer_id} ok: {message}")

    def end_step_error(self, layer_id: LayerId, message: str) -> None:
        """Description: Mark step error.
        Layer: L2
        Input: layer id
        Output: steps update
        """
        for s in reversed(self.steps):
            if s.layer_id == layer_id and s.status == "running":
                s.status = "error"
                s.finished_at_utc = utc_now_iso()
                s.message = message
                break
        self.feed(layer_id, "orchestrator", f"{layer_id} error: {message}")

    def touch(self) -> None:
        """Refresh state timestamp for legacy services."""
        self.updated_at_utc = utc_now_iso()

    def add_artifact(self, key: str, path: str, content_type: str = "application/json") -> None:
        """Add/overwrite artifact reference used by engine services."""
        self.artifacts[key] = ArtifactRef(path=path, mime=content_type)
        self.touch()

    def end_step(self, step_id: str, *, status: str = "ok", output_ref: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> None:
        """Legacy-compatible step finalizer keyed by agent message or id."""
        for s in reversed(self.steps):
            if s.status == "running" and (s.message == step_id or s.agent == step_id or s.layer_id.lower() in step_id.lower()):
                s.status = "blocked" if status == "blocked" else ("ok" if status == "ok" else "error")
                s.finished_at_utc = utc_now_iso()
                s.message = message or s.message
                break
        if message:
            self.feed(self.current_layer or "L0", "orchestrator", message)
        if output_ref:
            self.meta.setdefault("step_outputs", {})[step_id] = output_ref
        self.touch()

    def start_step(self, *args: Any, **kwargs: Any) -> None:
        """Start step supporting both new and legacy signatures."""
        if args and isinstance(args[0], str) and kwargs.get("layer_id"):
            # legacy style: start_step(step_id, layer_id=..., tool_name=..., input_ref=...)
            step_id = args[0]
            layer_id = kwargs.get("layer_id")
            agent = kwargs.get("tool_name", "agent")
            message = f"{step_id} started"
            progress = PROGRESS_MAP.get(str(layer_id), self.progress_percent)
        else:
            layer_id = args[0] if args else kwargs.get("layer_id")
            agent = (args[1] if len(args) > 1 else kwargs.get("agent", "agent"))
            message = (args[2] if len(args) > 2 else kwargs.get("message", "started"))
            progress = int(args[3] if len(args) > 3 else kwargs.get("progress", PROGRESS_MAP.get(str(layer_id), self.progress_percent)))

        self.current_layer = str(layer_id)
        self.progress_percent = int(progress)
        st = StepTrace(layer_id=layer_id, status="running", agent=str(agent), started_at_utc=utc_now_iso(), message=str(message))
        self.steps.append(st)
        self.feed(str(layer_id), str(agent), str(message))
        self.touch()



PROGRESS_MAP: Dict[str, int] = {
    "L0": 0,
    "L1": 10,
    "L2": 20,
    "L3": 40,
    "L4": 50,
    "L5": 60,
    "L6": 75,
    "L7": 85,
    "L8": 95,
    "L9": 100,
}
