from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


def _utc_now() -> datetime:
    """Description: Get current UTC timestamp.
    Layer: L0
    Input: None
    Output: datetime (UTC)
    """
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    """Description: Convert datetime to ISO-8601 Zulu time.
    Layer: L0
    Input: datetime
    Output: str (e.g., 2026-02-20T12:34:56Z)
    """
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


RunStatus = Literal["initialized", "running", "blocked", "needs_human_approval", "completed", "failed", "api_failure"]
class ArtifactRef(BaseModel):
    """Description: Reference to a stored artifact produced by an agent or tool.
    Layer: L1
    Input: Runtime outputs from any layer
    Output: Stable pointer used by downstream layers
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    path: str
    content_type: Optional[str] = None
    sha256: Optional[str] = None


class StepTrace(BaseModel):
    """Description: Audit trace for a single orchestration step.
    Layer: L2
    Input: Tool invocation metadata
    Output: Step record in OrchestrationState.steps
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str
    layer_id: str = "L2"
    tool_name: str = ""
    status: str = "running"
    message: Any = ""

    started_at_utc: str = Field(default_factory=lambda: _iso_utc(_utc_now()))
    finished_at_utc: Optional[str] = None

    input_ref: Dict[str, Any] = Field(default_factory=dict)
    output_ref: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_ref", mode="before")
    @classmethod
    def _coerce_input_ref(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {"input": v}

    @field_validator("output_ref", mode="before")
    @classmethod
    def _coerce_output_ref(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {"output": v}


class ApprovalDecision(BaseModel):
    """Description: Human approval record for sensitive steps.
    Layer: L5
    Input: Human decision + reason
    Output: Approval record stored in OrchestrationState.approvals
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    approved: bool
    reason: Optional[str] = None
    decided_by: Optional[str] = None
    decided_at_utc: str = Field(default_factory=lambda: _iso_utc(_utc_now()))


class InterviewChanceWeights(BaseModel):
    """Description: Weights for Interview Chance Score components.
    Layer: L4
    Input: Configured weights (w1,w2,w3)
    Output: Validated/normalized weights used for deterministic scoring
    """

    model_config = ConfigDict(extra="forbid")

    w1_skill_overlap: float = 0.45
    w2_experience_alignment: float = 0.35
    w3_ats_score: float = 0.20

    @model_validator(mode="after")
    def _validate_weights(self) -> "InterviewChanceWeights":
        for name, v in (
            ("w1_skill_overlap", self.w1_skill_overlap),
            ("w2_experience_alignment", self.w2_experience_alignment),
            ("w3_ats_score", self.w3_ats_score),
        ):
            if v < 0:
                raise ValueError(f"{name} must be >= 0")
        s = self.w1_skill_overlap + self.w2_experience_alignment + self.w3_ats_score
        if s <= 0:
            raise ValueError("At least one weight must be > 0")

        # Normalize to sum=1 for stability.
        self.w1_skill_overlap /= s
        self.w2_experience_alignment /= s
        self.w3_ats_score /= s
        return self


class InterviewChanceComponents(BaseModel):
    """Description: Normalized components for interview chance scoring.
    Layer: L4
    Input: Component scores computed by match/eval agents
    Output: Deterministic inputs for InterviewChanceScore
    """

    model_config = ConfigDict(extra="forbid")

    skill_overlap: float = 0.0
    experience_alignment: float = 0.0
    ats_score: float = 0.0
    market_competition_factor: float = 1.0  # penalty, >= 1.0

    @field_validator("skill_overlap", "experience_alignment", "ats_score")
    @classmethod
    def _bounded_01(cls, v: float) -> float:
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError("score components must be in [0, 1]")
        return float(v)

    @field_validator("market_competition_factor")
    @classmethod
    def _market_factor(cls, v: float) -> float:
        v = float(v)
        if v < 1.0:
            raise ValueError("market_competition_factor must be >= 1.0")
        return v


class InterviewChanceBreakdown(BaseModel):
    """Description: Deterministic interview chance score breakdown.
    Layer: L4
    Input: Weights + components
    Output: Computed InterviewChanceScore in [0,1]
    """

    model_config = ConfigDict(extra="forbid")

    weights: InterviewChanceWeights = Field(default_factory=InterviewChanceWeights)
    components: InterviewChanceComponents = Field(default_factory=InterviewChanceComponents)

    @computed_field
    @property
    def interview_chance_score(self) -> float:
        """Description: Compute normalized Interview Chance Score.
        Layer: L4
        Input: weights + components
        Output: float in [0,1]
        """
        base = (
            self.weights.w1_skill_overlap * self.components.skill_overlap
            + self.weights.w2_experience_alignment * self.components.experience_alignment
            + self.weights.w3_ats_score * self.components.ats_score
        )
        # MarketCompetitionFactor is a penalty (>=1): higher competition lowers the score.
        return max(0.0, min(1.0, base / self.components.market_competition_factor))


class EvaluationEvent(BaseModel):
    """Description: Evaluation result used by the Recursive Gate pattern.
    Layer: L3-L7
    Input: Generator output reference + evaluator metrics
    Output: Stored evaluation record + gate decision inputs
    """

    model_config = ConfigDict(extra="forbid")

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

    started_at_utc: str = Field(default_factory=lambda: _iso_utc(_utc_now()))
    finished_at_utc: str = Field(default_factory=lambda: _iso_utc(_utc_now()))

    @field_validator("evaluation_score", "threshold")
    @classmethod
    def _bounded_01(cls, v: float) -> float:
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError("evaluation_score/threshold must be in [0,1]")
        return float(v)

    @computed_field
    @property
    def passed(self) -> bool:
        return self.evaluation_score >= self.threshold

    def should_retry(self) -> bool:
        """Description: Recursive Gate decision helper.
        Layer: L3
        Input: EvaluationEvent
        Output: bool indicating if loop-back is permitted
        """
        return (not self.passed) and (self.retry_count < self.max_retries)


class OrchestrationState(BaseModel):
    """Description: The heart of CareerAgent-AI runtime state.

    Layer: L2
    Input: New run request from API/UI
    Output: Stateful, auditable object passed through LangGraph nodes
    """

    model_config = ConfigDict(extra="forbid")

    version: str = "v1"
    run_id: str = Field(default_factory=lambda: uuid4().hex)

    created_at_utc: str = Field(default_factory=lambda: _iso_utc(_utc_now()))
    updated_at_utc: str = Field(default_factory=lambda: _iso_utc(_utc_now()))

    status: RunStatus = "initialized"
    mode: str = "agentic"
    env: Optional[str] = None
    git_sha: Optional[str] = None

    artifacts: Dict[str, ArtifactRef] = Field(default_factory=dict)
    steps: List[StepTrace] = Field(default_factory=list)
    approvals: List[ApprovalDecision] = Field(default_factory=list)
    evaluations: List[EvaluationEvent] = Field(default_factory=list)

    meta: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(cls, *, env: Optional[str] = None, mode: str = "agentic", git_sha: Optional[str] = None) -> "OrchestrationState":
        """Description: Create a new orchestration run.
        Layer: L2
        Input: env/mode/git_sha from L0 config
        Output: Initialized OrchestrationState
        """
        st = cls(env=env, mode=mode, git_sha=git_sha)
        st.status = "running"
        st.updated_at_utc = _iso_utc(_utc_now())
        return st

    def touch(self) -> None:
        """Description: Update updated_at_utc timestamp.
        Layer: L2
        Input: Internal
        Output: None
        """
        self.updated_at_utc = _iso_utc(_utc_now())

    def add_artifact(self, key: str, path: str, *, content_type: Optional[str] = None, sha256: Optional[str] = None) -> ArtifactRef:
        """Description: Register an artifact reference.
        Layer: L2
        Input: Artifact key + path
        Output: ArtifactRef stored in state
        """
        ref = ArtifactRef(key=key, path=path, content_type=content_type, sha256=sha256)
        self.artifacts[key] = ref
        self.touch()
        return ref

    def start_step(self, step_id: str, *, layer_id: str = "L2", tool_name: str = "", input_ref: Optional[Dict[str, Any]] = None) -> StepTrace:
        """Description: Start a step trace entry.
        Layer: L2
        Input: Step metadata + input references
        Output: StepTrace appended to state
        """
        tr = StepTrace(step_id=step_id, layer_id=layer_id, tool_name=tool_name, status="running", input_ref=input_ref or {})
        self.steps.append(tr)
        self.touch()
        return tr

    def end_step(self, step_id: str, *, status: str = "ok", output_ref: Optional[Dict[str, Any]] = None, message: Any = "") -> StepTrace:
        """Description: Complete a step trace entry.
        Layer: L2
        Input: Step id + outputs
        Output: Updated StepTrace
        """
        target: Optional[StepTrace] = None
        for s in reversed(self.steps):
            if s.step_id == step_id:
                target = s
                break
        if target is None:
            target = self.start_step(step_id)

        target.status = status
        target.message = message
        target.finished_at_utc = _iso_utc(_utc_now())
        target.output_ref = output_ref or {}
        self.touch()
        return target

    def record_approval(self, tool_name: str, approved: bool, *, reason: Optional[str] = None, decided_by: Optional[str] = None) -> ApprovalDecision:
        """Description: Record a human approval decision.
        Layer: L5
        Input: Tool name + decision
        Output: ApprovalDecision stored in state
        """
        d = ApprovalDecision(tool_name=tool_name, approved=approved, reason=reason, decided_by=decided_by)
        self.approvals.append(d)
        self.touch()
        return d

    def is_approved(self, tool_name: str) -> bool:
        """Description: Resolve latest approval decision for a tool.
        Layer: L5
        Input: Tool name
        Output: bool
        """
        for d in reversed(self.approvals):
            if d.tool_name == tool_name:
                return bool(d.approved)
        return False

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
        """Description: Append an evaluation event for the Recursive Gate.
        Layer: L3
        Input: Evaluation metadata + scores
        Output: EvaluationEvent stored in state
        """
        ev = EvaluationEvent(
            layer_id=layer_id,
            target_id=target_id,
            generator_agent=generator_agent,
            evaluator_agent=evaluator_agent,
            evaluation_score=evaluation_score,
            threshold=threshold,
            feedback=feedback or [],
            retry_count=retry_count,
            max_retries=max_retries,
            interview_chance=interview_chance,
        )
        self.evaluations.append(ev)
        self.touch()
        return ev

    def latest_evaluation(self, *, target_id: str, layer_id: Optional[str] = None) -> Optional[EvaluationEvent]:
        """Description: Fetch the most recent evaluation for a target.
        Layer: L3
        Input: target_id, optional layer filter
        Output: EvaluationEvent | None
        """
        for ev in reversed(self.evaluations):
            if ev.target_id != target_id:
                continue
            if layer_id and ev.layer_id != layer_id:
                continue
            return ev
        return None

    def apply_recursive_gate(self, *, target_id: str, layer_id: str) -> Literal["pass", "retry", "human_approval"]:
        """Description: Decide next action for the Recursive Gate.
        Layer: L3
        Input: Latest EvaluationEvent for target
        Output: pass|retry|human_approval
        """
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
