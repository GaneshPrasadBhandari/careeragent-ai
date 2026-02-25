from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from careeragent.core.state import AgentState
from careeragent.agents.matcher_agent_schema import MatchReport, JobDescription
from careeragent.agents.parser_agent_service import ExtractedResume
from careeragent.agents.strategy_agent_schema import ActionItem, PivotStrategy


class _StrategyGraphState(TypedDict):
    """
    Description: LangGraph state for L5 strategy generation.
    Layer: L5
    Input: resume + job + match_report + feedback
    Output: PivotStrategy
    """

    resume: ExtractedResume
    job: JobDescription
    match_report: MatchReport
    feedback: List[str]
    orchestration_state: AgentState
    strategy: Optional[PivotStrategy]


class StrategyAgentService:
    """
    Description: L5 strategist that generates pivot strategy + action items.
    Layer: L5
    Input: MatchReport + Resume + JobDescription
    Output: PivotStrategy
    """

    def as_runnable(self) -> RunnableLambda:
        """
        Description: Expose strategist as runnable.
        Layer: L5
        Input: dict payload
        Output: PivotStrategy
        """
        def _run(payload: Dict[str, Any]) -> PivotStrategy:
            return self.generate(
                resume=payload["resume"],
                job=payload["job"],
                match_report=payload["match_report"],
                orchestration_state=payload["orchestration_state"],
                feedback=payload.get("feedback") or [],
            )
        return RunnableLambda(_run)

    def build_langgraph(self) -> Any:
        """
        Description: Build minimal LangGraph for strategy.
        Layer: L5
        Input: None
        Output: Compiled graph runnable
        """
        g = StateGraph(_StrategyGraphState)

        def _node(state: _StrategyGraphState) -> _StrategyGraphState:
            state["strategy"] = self.generate(
                resume=state["resume"],
                job=state["job"],
                match_report=state["match_report"],
                orchestration_state=state["orchestration_state"],
                feedback=state.get("feedback") or [],
            )
            return state

        g.add_node("strategy", _node)
        g.set_entry_point("strategy")
        g.add_edge("strategy", END)
        return g.compile()

    def generate(
        self,
        *,
        resume: ExtractedResume,
        job: JobDescription,
        match_report: MatchReport,
        orchestration_state: AgentState,
        feedback: Optional[List[str]] = None,
    ) -> PivotStrategy:
        """
        Description: Generate a pivot strategy if match < 70%, else optimization plan.
        Layer: L5
        Input: resume + job + match_report + feedback
        Output: PivotStrategy
        """
        fb = [f.strip() for f in (feedback or []) if f and str(f).strip()]
        m = float(match_report.overall_match_percent)

        if m >= 85:
            posture = "proceed"
        elif m >= 70:
            posture = "proceed_with_edits"
        else:
            posture = "pivot"

        items: List[ActionItem] = []

        # Default behavior: keep it concise; evaluator may request more depth
        want_more = any("more" in x.lower() or "add" in x.lower() for x in fb)

        if posture == "pivot":
            items.extend(self._pivot_items(match_report, want_more=want_more))
        else:
            items.extend(self._optimize_items(match_report, want_more=want_more))

        return PivotStrategy(
            job_id=match_report.job_id,
            overall_match_percent=m,
            posture=posture,
            action_items=items,
        )

    @staticmethod
    def _pivot_items(match_report: MatchReport, *, want_more: bool) -> List[ActionItem]:
        """
        Description: Generate pivot action items.
        Layer: L5
        Input: MatchReport
        Output: list[ActionItem]
        """
        gaps = match_report.missing_required_skills[:8]
        base = [
            ActionItem(
                title="Reframe experience around the role’s core outcomes",
                why_it_matters="Hiring managers screen for outcome-aligned evidence, not just titles.",
                how_to_execute=[
                    "Rewrite your Summary to mirror the job’s top 3 responsibilities (only what you’ve actually done).",
                    "Move the most relevant project/experience bullets to the top of Experience.",
                    "Add measurable impact (latency, cost reduction, adoption, accuracy, revenue).",
                ],
                priority="high",
                eta_days=1,
            ),
            ActionItem(
                title="Close skill gaps with proof-based micro-projects",
                why_it_matters="If you’re missing required skills, you need evidence fast—not claims.",
                how_to_execute=[
                    f"Pick 1–2 gaps and build a small repo that demonstrates them: {', '.join(gaps) if gaps else 'top gaps'}",
                    "Add a short 'Projects' section with 2 bullets: what you built + what metric improved.",
                    "Link GitHub in contact links and include the repo in your cover letter.",
                ],
                priority="high",
                eta_days=3,
            ),
        ]
        if want_more:
            base.append(
                ActionItem(
                    title="Keyword-map your existing skills to the job language",
                    why_it_matters="ATS and recruiters search by the job’s vocabulary; synonyms can hide relevance.",
                    how_to_execute=[
                        "Create a 2-column mapping: Job keyword → your equivalent experience evidence.",
                        "Update Skills section with exact job keywords (only if true).",
                        "Add 1 bullet per mapped keyword under the most relevant experience item.",
                    ],
                    priority="medium",
                    eta_days=1,
                )
            )
        return base

    @staticmethod
    def _optimize_items(match_report: MatchReport, *, want_more: bool) -> List[ActionItem]:
        """
        Description: Generate optimization items when match is moderate/high.
        Layer: L5
        Input: MatchReport
        Output: list[ActionItem]
        """
        base = [
            ActionItem(
                title="Increase skill-matching density in the top half of the resume",
                why_it_matters="Recruiters often decide in <30 seconds; top placement boosts interview probability.",
                how_to_execute=[
                    "Move the most relevant skills to the first line of Skills.",
                    "Ensure the first 2 experience bullets contain 2–3 job keywords each (only if true).",
                ],
                priority="high",
                eta_days=1,
            ),
            ActionItem(
                title="Turn rationale gaps into targeted edits",
                why_it_matters="Your MatchReport already tells you what’s missing; use it as an edit checklist.",
                how_to_execute=[
                    f"Address missing required skills through evidence or learning plan: {', '.join(match_report.missing_required_skills[:6]) or 'None'}",
                    "Add 1 quantified metric to each of your top 3 bullets.",
                ],
                priority="medium",
                eta_days=2,
            ),
        ]
        if want_more:
            base.append(
                ActionItem(
                    title="Build a job-specific 30-second positioning statement",
                    why_it_matters="This improves cover letter, recruiter calls, and interviews simultaneously.",
                    how_to_execute=[
                        "Write: 'I help <domain> achieve <outcome> using <tools>, proven by <metric>.'",
                        "Use the same structure in resume Summary + cover letter opening.",
                    ],
                    priority="medium",
                    eta_days=1,
                )
            )
        return base
