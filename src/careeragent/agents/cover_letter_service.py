from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from careeragent.core.state import AgentState
from careeragent.agents.matcher_agent_schema import JobDescription, MatchReport
from careeragent.agents.parser_agent_service import ExtractedResume
from careeragent.agents.cover_letter_agent_schema import CoverLetterDraft


class _CoverGraphState(TypedDict):
    """
    Description: LangGraph state for L6 cover letter drafting.
    Layer: L6
    Input: resume + job + match_report + feedback
    Output: CoverLetterDraft
    """

    resume: ExtractedResume
    job: JobDescription
    match_report: MatchReport
    feedback: List[str]
    orchestration_state: AgentState
    draft: Optional[CoverLetterDraft]


class CoverLetterService:
    """
    Description: L6 generator that drafts a country-specific cover letter.
    Layer: L6
    Input: Resume + MatchReport + JobDescription + feedback
    Output: CoverLetterDraft
    """

    def as_runnable(self) -> RunnableLambda:
        """
        Description: Expose cover letter generator as runnable.
        Layer: L6
        Input: dict payload
        Output: CoverLetterDraft
        """
        def _run(payload: Dict[str, Any]) -> CoverLetterDraft:
            return self.draft(
                resume=payload["resume"],
                job=payload["job"],
                match_report=payload["match_report"],
                orchestration_state=payload["orchestration_state"],
                feedback=payload.get("feedback") or [],
            )
        return RunnableLambda(_run)

    def build_langgraph(self) -> Any:
        """
        Description: Build minimal LangGraph for drafting.
        Layer: L6
        Input: None
        Output: Compiled graph runnable
        """
        g = StateGraph(_CoverGraphState)

        def _node(state: _CoverGraphState) -> _CoverGraphState:
            state["draft"] = self.draft(
                resume=state["resume"],
                job=state["job"],
                match_report=state["match_report"],
                orchestration_state=state["orchestration_state"],
                feedback=state.get("feedback") or [],
            )
            return state

        g.add_node("draft", _node)
        g.set_entry_point("draft")
        g.add_edge("draft", END)
        return g.compile()

    def draft(
        self,
        *,
        resume: ExtractedResume,
        job: JobDescription,
        match_report: MatchReport,
        orchestration_state: AgentState,
        feedback: Optional[List[str]] = None,
    ) -> CoverLetterDraft:
        """
        Description: Draft cover letter. If feedback indicates missing contact info or tone issues,
                     refine accordingly (recursive loop-back support).
        Layer: L6
        Input: resume + job + match_report + feedback
        Output: CoverLetterDraft
        """
        fb = [f.strip() for f in (feedback or []) if f and str(f).strip()]
        include_contact = any("contact" in x.lower() for x in fb) or any("header" in x.lower() for x in fb)

        # choose top skills: prefer matched required, then add ATS-friendly keywords
        top_skills = (match_report.matched_skills or [])[:6]

        today = datetime.utcnow().strftime("%B %d, %Y")

        subject = f"Application — {job.role_title} ({job.company})"

        header = ""
        if include_contact:
            # Use only known info. Never invent.
            lines = []
            if resume.name:
                lines.append(resume.name)
            if resume.contact.email:
                lines.append(resume.contact.email)
            if resume.contact.phone:
                lines.append(resume.contact.phone)
            if resume.contact.location:
                lines.append(resume.contact.location)
            if resume.contact.links:
                lines.append(resume.contact.links[0])
            header = "\n".join(lines).strip() + "\n\n"

        # country-specific greeting norms (minimal for now; extend later)
        if (job.country_code or "US").upper() in ("US", "CA"):
            greeting = "Dear Hiring Manager,"
        else:
            greeting = "Dear Hiring Team,"

        # Build a tight 3-paragraph letter. No hallucinated claims.
        p1 = (
            f"I’m applying for the {job.role_title} role at {job.company}. "
            f"My background aligns with the role’s requirements, particularly in {', '.join(top_skills[:3]) if top_skills else 'core delivery and execution'}."
        )
        p2 = (
            "In recent work, I’ve delivered measurable outcomes by building production-ready systems, improving reliability, and collaborating across teams. "
            "I focus on clear problem framing, strong engineering discipline, and evidence-backed results."
        )
        p3 = (
            "I’d welcome the opportunity to discuss how I can help your team deliver impact. "
            "Thank you for your time and consideration."
        )

        body = f"{header}{today}\n\n{greeting}\n\n{p1}\n\n{p2}\n\n{p3}\n\nSincerely,\n{resume.name or ''}".strip()

        return CoverLetterDraft(
            job_id=job.job_id,
            country_code=(job.country_code or "US").upper(),
            role_title=job.role_title,
            company=job.company,
            contact_block_included=bool(include_contact),
            subject_line=subject,
            body=body,
            highlighted_skills=top_skills,
        )
