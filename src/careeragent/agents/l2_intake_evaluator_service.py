from __future__ import annotations

from typing import List, Tuple

from careeragent.core.state import AgentState, EvaluationEntry


class L2IntakeEvaluatorService:
    """Description: Mandatory intake quality gate.
    Layer: L2
    Input: extracted_profile
    Output: evaluation entry + decision
    """

    def evaluate(self, state: AgentState) -> Tuple[float, str, List[str]]:
        fb: List[str] = []
        prof = state.extracted_profile or {}
        contact = (prof.get("contact") or {})

        if not prof.get("name"):
            fb.append("Name missing; consider adding full name at top of resume.")
        if not contact.get("email"):
            fb.append("Email missing; add a valid email address.")
        if not contact.get("phone"):
            fb.append("Phone missing; add a US phone number.")
        skills = prof.get("skills") or []
        if len(skills) < 10:
            fb.append("Few skills extracted; ensure a Key Skills section with comma-separated tools.")
        exp = prof.get("experience") or []
        if len(exp) == 0:
            fb.append("Experience section not parsed; ensure clear job titles and bullet points.")

        score = 1.0
        if fb:
            # soft score
            score = max(0.45, 1.0 - (0.12 * len(fb)))

        decision = "pass" if score >= state.preferences.resume_threshold else "retry"
        reason = "intake ok" if decision == "pass" else "intake needs cleanup"
        return score, reason, fb

    def write_to_state(self, state: AgentState, score: float, reason: str, feedback: List[str]) -> None:
        entry = EvaluationEntry(
            layer_id="L2",
            target_id="intake_bundle",
            evaluation_score=float(score),
            threshold=float(state.preferences.resume_threshold),
            decision="pass" if score >= state.preferences.resume_threshold else "retry",
            feedback=feedback,
        )
        state.evaluations.append(entry)
        state.log_eval(f"[L2] {reason} score={score:.2f} feedback={len(feedback)}")
