from __future__ import annotations

from uuid import uuid4
from typing import Any, Dict, Optional, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from careeragent.core.state import AgentState, _iso_utc, _utc_now
from careeragent.agents.apply_executor_schema import ApplicationSubmission


class _ApplyGraphState(TypedDict):
    """
    Description: LangGraph state for L7 application submission.
    Layer: L7
    Input: state + artifact keys
    Output: ApplicationSubmission
    """

    orchestration_state: AgentState
    job_id: str
    resume_artifact_key: str
    cover_letter_artifact_key: str
    submission: Optional[ApplicationSubmission]


class ApplyExecutorService:
    """
    Description: L7 executor that simulates an "Application Submit" action.
    Layer: L7
    Input: Final resume + cover letter artifact keys and job_id
    Output: ApplicationSubmission recorded into AgentState
    """

    def as_runnable(self) -> RunnableLambda:
        """
        Description: Expose apply executor as a LangChain runnable.
        Layer: L7
        Input: dict(orchestration_state, job_id, resume_artifact_key, cover_letter_artifact_key)
        Output: ApplicationSubmission
        """
        def _run(payload: Dict[str, Any]) -> ApplicationSubmission:
            return self.submit(
                orchestration_state=payload["orchestration_state"],
                job_id=payload["job_id"],
                resume_artifact_key=payload["resume_artifact_key"],
                cover_letter_artifact_key=payload["cover_letter_artifact_key"],
                notes=payload.get("notes"),
            )
        return RunnableLambda(_run)

    def build_langgraph(self) -> Any:
        """
        Description: Build minimal LangGraph for application submission.
        Layer: L7
        Input: None
        Output: Compiled graph runnable
        """
        g = StateGraph(_ApplyGraphState)

        def _node(state: _ApplyGraphState) -> _ApplyGraphState:
            state["submission"] = self.submit(
                orchestration_state=state["orchestration_state"],
                job_id=state["job_id"],
                resume_artifact_key=state["resume_artifact_key"],
                cover_letter_artifact_key=state["cover_letter_artifact_key"],
            )
            return state

        g.add_node("submit", _node)
        g.set_entry_point("submit")
        g.add_edge("submit", END)
        return g.compile()

    def submit(
        self,
        *,
        orchestration_state: AgentState,
        job_id: str,
        resume_artifact_key: str,
        cover_letter_artifact_key: str,
        notes: Optional[str] = None,
    ) -> ApplicationSubmission:
        """
        Description: Simulate an application submission and record submission_id + timestamp in state.
        Layer: L7
        Input: AgentState + job_id + artifact keys
        Output: ApplicationSubmission
        """
        submission_id = uuid4().hex
        submitted_at_utc = _iso_utc(_utc_now())

        submission = ApplicationSubmission(
            submission_id=submission_id,
            job_id=str(job_id),
            resume_artifact_key=str(resume_artifact_key),
            cover_letter_artifact_key=str(cover_letter_artifact_key),
            submitted_at_utc=submitted_at_utc,
            notes=notes,
        )

        # Record in state meta for easy cross-layer joins (analytics).
        orchestration_state.meta.setdefault("submissions", {})
        orchestration_state.meta["submissions"][submission_id] = submission.model_dump()

        # IMPORTANT: RunStatus becomes completed ONLY after L7 success.
        orchestration_state.status = "completed"
        orchestration_state.touch()

        return submission
