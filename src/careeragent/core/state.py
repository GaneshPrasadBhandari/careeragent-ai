from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


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
    status: Literal["queued", "running", "ok", "error"] = "queued"
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


class InterviewChanceWeights(BaseModel):
    """Weighted blend for interview chance scoring."""

    w1_skill_overlap: float = 0.45
    w2_experience_alignment: float = 0.35
    w3_ats_score: float = 0.20


class InterviewChanceComponents(BaseModel):
    """Normalized L4 components."""

    skill_overlap: float = 0.0
    experience_alignment: float = 0.0
    ats_score: float = 0.0
    market_competition_factor: float = 1.0


class InterviewChanceBreakdown(BaseModel):
    """Final interview chance deterministic breakdown."""

    weights: InterviewChanceWeights = Field(default_factory=InterviewChanceWeights)
    components: InterviewChanceComponents = Field(default_factory=InterviewChanceComponents)

    @property
    def interview_chance_score(self) -> float:
        base = (
            self.weights.w1_skill_overlap * self.components.skill_overlap
            + self.weights.w2_experience_alignment * self.components.experience_alignment
            + self.weights.w3_ats_score * self.components.ats_score
        )
        m = max(1.0, float(self.components.market_competition_factor or 1.0))
        return max(0.0, min(1.0, base / m))


class EvaluationEvent(BaseModel):
    """Recursive-Gate evaluation event audit record."""

    eval_id: str = Field(default_factory=lambda: uuid4().hex)
    layer_id: str
    target_id: str
    generator_agent: str
    evaluator_agent: str
    evaluation_score: float
    threshold: float
    feedback: List[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    interview_chance: Optional[InterviewChanceBreakdown] = None
    started_at_utc: str = Field(default_factory=utc_now_iso)
    finished_at_utc: str = Field(default_factory=utc_now_iso)

    @property
    def passed(self) -> bool:
        return float(self.evaluation_score) >= float(self.threshold)

    def should_retry(self) -> bool:
        return (not self.passed) and int(self.retry_count) < int(self.max_retries)


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
    resume_threshold: float = 0.55
    draft_count: int = 10

    user_phone: Optional[str] = None
    linkedin_url: Optional[str] = None


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

    status: Literal["queued", "running", "needs_human_approval", "completed", "failed"] = "queued"
    pending_action: Optional[str] = None

    current_layer: Optional[str] = None
    progress_percent: int = 0

    preferences: Preferences = Field(default_factory=Preferences)

    # --- Outputs
    extracted_profile: Dict[str, Any] = Field(default_factory=dict)
    jobs_raw: List[Dict[str, Any]] = Field(default_factory=list)
    jobs_scored: List[Dict[str, Any]] = Field(default_factory=list)
    ranking: List[Dict[str, Any]] = Field(default_factory=list)

    approved_job_urls: List[str] = Field(default_factory=list)

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
    evaluations: List[Any] = Field(default_factory=list)
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

    def touch(self) -> None:
        """Compatibility no-op timestamp hook used by legacy services."""
        self.meta["updated_at_utc"] = utc_now_iso()

    def start_step(self, layer_id: LayerId, agent: str, message: str, progress: int) -> None:
        """Description: Start a step and update progress.
        Layer: L2
        Input: layer id, agent
        Output: steps + progress
        """
        self.current_layer = layer_id
        self.progress_percent = int(progress)
        st = StepTrace(layer_id=layer_id, status="running", agent=agent, started_at_utc=utc_now_iso(), message=message)
        # replace any existing same layer running
        self.steps = [s for s in self.steps if not (s.layer_id == layer_id and s.status == "running")]
        self.steps.append(st)
        self.feed(layer_id, agent, message)

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

    def record_evaluation(
        self,
        *,
        layer_id: str,
        target_id: str,
        generator_agent: str,
        evaluator_agent: str,
        evaluation_score: float,
        threshold: float,
        feedback: Optional[List[str]] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        interview_chance: Optional[InterviewChanceBreakdown] = None,
    ) -> EvaluationEvent:
        ev = EvaluationEvent(
            layer_id=layer_id,
            target_id=target_id,
            generator_agent=generator_agent,
            evaluator_agent=evaluator_agent,
            evaluation_score=float(evaluation_score),
            threshold=float(threshold),
            feedback=feedback or [],
            retry_count=int(retry_count),
            max_retries=int(max_retries),
            interview_chance=interview_chance,
        )
        self.evaluations.append(ev)
        self.touch()
        return ev

    def latest_evaluation(self, *, target_id: str, layer_id: Optional[str] = None) -> Optional[EvaluationEvent]:
        for ev in reversed(self.evaluations):
            if not isinstance(ev, EvaluationEvent):
                continue
            if ev.target_id != target_id:
                continue
            if layer_id and ev.layer_id != layer_id:
                continue
            return ev
        return None

    def apply_recursive_gate(self, *, target_id: str, layer_id: str) -> Literal["pass", "retry", "human_approval"]:
        ev = self.latest_evaluation(target_id=target_id, layer_id=layer_id)
        if ev is None:
            return "retry"
        if ev.passed:
            return "pass"
        if ev.should_retry():
            return "retry"
        self.status = "needs_human_approval"
        self.touch()
        return "human_approval"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
