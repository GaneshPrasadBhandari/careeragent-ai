from __future__ import annotations

from uuid import uuid4
from typing import Dict, List, Optional

from careeragent.orchestration.state import AgentState, _iso_utc, _utc_now
from careeragent.agents.application_tracker_schema import ApplicationStatus, StatusUpdateEvent


class ApplicationTrackerService:
    """
    Description: L8 tracker that records and monitors application statuses.
    Layer: L8
    Input: AgentState + submission_id + status
    Output: StatusUpdateEvent list stored in AgentState.meta
    """

    _ALLOWED_TRANSITIONS: Dict[ApplicationStatus, List[ApplicationStatus]] = {
        "applied": ["interviewing", "rejected"],
        "interviewing": ["rejected"],  # extend later: offered/accepted
        "rejected": [],
    }

    def record_status_update(
        self,
        *,
        orchestration_state: AgentState,
        submission_id: str,
        job_id: str,
        new_status: ApplicationStatus,
        note: Optional[str] = None,
    ) -> StatusUpdateEvent:
        """
        Description: Record a status update event with transition validation.
        Layer: L8
        Input: state + submission_id + job_id + new_status
        Output: StatusUpdateEvent
        """
        # transition validation (best-effort; does not hard-fail runs)
        current = self.get_current_status(orchestration_state=orchestration_state, submission_id=submission_id)
        if current is not None:
            allowed = self._ALLOWED_TRANSITIONS.get(current, [])
            if new_status not in allowed and new_status != current:
                # Log a warning-like note into meta for audit visibility.
                orchestration_state.meta.setdefault("tracker_warnings", [])
                orchestration_state.meta["tracker_warnings"].append(
                    f"Invalid transition attempted for {submission_id}: {current} -> {new_status}"
                )

        ev = StatusUpdateEvent(
            event_id=uuid4().hex,
            submission_id=str(submission_id),
            job_id=str(job_id),
            status=new_status,
            occurred_at_utc=_iso_utc(_utc_now()),
            note=note,
        )

        orchestration_state.meta.setdefault("status_updates", [])
        orchestration_state.meta["status_updates"].append(ev.model_dump())
        orchestration_state.touch()
        return ev

    def get_current_status(self, *, orchestration_state: AgentState, submission_id: str) -> Optional[ApplicationStatus]:
        """
        Description: Return the most recent status for a submission.
        Layer: L8
        Input: state + submission_id
        Output: ApplicationStatus | None
        """
        events = orchestration_state.meta.get("status_updates", []) or []
        for e in reversed(events):
            if e.get("submission_id") == submission_id:
                return e.get("status")
        return None
