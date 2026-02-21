from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from careeragent.orchestration.state import (
    InterviewChanceBreakdown,
    InterviewChanceComponents,
    InterviewChanceWeights,
    OrchestrationState,
)
from careeragent.agents.parser_agent_service import ExtractedResume
from careeragent.agents.matcher_agent_schema import JobDescription, MatchComponents, MatchReport


class _MatcherGraphState(TypedDict):
    """
    Description: LangGraph state for L4 matching.
    Layer: L4
    Input: resume + job + orchestration_state
    Output: MatchReport
    """

    resume: ExtractedResume
    job: JobDescription
    orchestration_state: OrchestrationState
    report: Optional[MatchReport]


class MatcherAgentService:
    """
    Description: L4 generator that matches ExtractedResume to a JobDescription.
    Layer: L4
    Input: ExtractedResume + JobDescription JSON
    Output: MatchReport (with InterviewChanceScore)
    """

    def as_runnable(self) -> RunnableLambda:
        """
        Description: Expose matcher as a LangChain runnable.
        Layer: L4
        Input: dict(resume, job, orchestration_state)
        Output: MatchReport
        """
        def _run(payload: Dict[str, Any]) -> MatchReport:
            return self.match(
                resume=payload["resume"],
                job=payload["job"],
                orchestration_state=payload["orchestration_state"],
            )
        return RunnableLambda(_run)

    def build_langgraph(self) -> Any:
        """
        Description: Build minimal LangGraph graph for matching.
        Layer: L4
        Input: None
        Output: Compiled graph runnable
        """
        g = StateGraph(_MatcherGraphState)

        def _match_node(state: _MatcherGraphState) -> _MatcherGraphState:
            state["report"] = self.match(
                resume=state["resume"],
                job=state["job"],
                orchestration_state=state["orchestration_state"],
            )
            return state

        g.add_node("match", _match_node)
        g.set_entry_point("match")
        g.add_edge("match", END)
        return g.compile()

    def match(self, *, resume: ExtractedResume, job: JobDescription, orchestration_state: OrchestrationState) -> MatchReport:
        """
        Description: Compute deterministic match report + InterviewChanceScore.
        Layer: L4
        Input: ExtractedResume + JobDescription + OrchestrationState
        Output: MatchReport
        """
        resume_skills = self._norm_set(resume.skills)
        req_skills = self._norm_set(job.required_skills)
        pref_skills = self._norm_set(job.preferred_skills)

        matched_req = sorted(list(resume_skills.intersection(req_skills)))
        missing_req = sorted(list(req_skills.difference(resume_skills)))
        missing_pref = sorted(list(pref_skills.difference(resume_skills)))

        skill_overlap = self._skill_overlap(resume_skills, req_skills)
        exp_align = self._experience_alignment(resume, job)
        ats = self._ats_score(resume)

        market = self._market_factor(job)
        components = MatchComponents(
            skill_overlap=skill_overlap,
            experience_alignment=exp_align,
            ats_score=ats,
            market_competition_factor=market,
        )

        breakdown = self._interview_chance_breakdown(orchestration_state, components)
        interview = breakdown.interview_chance_score
        overall = round(interview * 100.0, 2)

        rationale = self._rationale(matched_req, missing_req, components)

        return MatchReport(
            job_id=job.job_id,
            role_title=job.role_title,
            company=job.company,
            matched_skills=matched_req,
            missing_required_skills=missing_req,
            missing_preferred_skills=missing_pref,
            components=components,
            interview_chance_score=float(interview),
            overall_match_percent=float(overall),
            rationale=rationale,
        )

    # ---------------- internals ----------------

    @staticmethod
    def _norm_set(items: List[str]) -> set[str]:
        """
        Description: Normalize strings into a lowercase set for overlap math.
        Layer: L4
        Input: list[str]
        Output: set[str]
        """
        out = set()
        for it in items or []:
            s = re.sub(r"\s+", " ", str(it).strip().lower())
            if s:
                out.add(s)
        return out

    @staticmethod
    def _skill_overlap(resume_skills: set[str], required_skills: set[str]) -> float:
        """
        Description: SkillOverlap = |intersection| / |required|.
        Layer: L4
        Input: resume skills set + required skills set
        Output: float in [0,1]
        """
        if not required_skills:
            return 0.0
        return max(0.0, min(1.0, len(resume_skills.intersection(required_skills)) / len(required_skills)))

    @staticmethod
    def _ats_score(resume: ExtractedResume) -> float:
        """
        Description: ATS score proxy from structural completeness and density.
        Layer: L4
        Input: ExtractedResume
        Output: float in [0,1]
        """
        # contact presence
        contact_ok = 1.0 if (resume.contact.email and (resume.contact.phone or resume.contact.links)) else 0.0
        # skills density
        skills_ok = min(1.0, len(resume.skills) / 12.0) if resume.skills else 0.0
        # experience bullet density
        bullets = 0
        for x in resume.experience or []:
            bullets += len(x.bullets or [])
        bullets_ok = min(1.0, bullets / 10.0) if bullets else 0.0

        score = (0.30 * contact_ok) + (0.35 * skills_ok) + (0.35 * bullets_ok)
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _experience_alignment(resume: ExtractedResume, job: JobDescription) -> float:
        """
        Description: ExperienceAlignment via cosine similarity between experience bullets and requirements_text.
        Layer: L4
        Input: ExtractedResume + JobDescription
        Output: float in [0,1]
        """
        req = (job.requirements_text or "").strip()
        exp = " ".join([" ".join(x.bullets or []) for x in (resume.experience or [])]).strip()
        return MatcherAgentService._cosine_sim(exp, req)

    @staticmethod
    def _cosine_sim(a: str, b: str) -> float:
        """
        Description: Lightweight cosine similarity without external ML deps.
        Layer: L4
        Input: two strings
        Output: similarity in [0,1]
        """
        a = (a or "").lower().strip()
        b = (b or "").lower().strip()
        if not a or not b:
            return 0.0

        def toks(s: str) -> List[str]:
            return [t for t in re.split(r"[^a-z0-9\+\.#]+", s) if t and len(t) > 1]

        ca = Counter(toks(a))
        cb = Counter(toks(b))
        common = set(ca).intersection(set(cb))
        dot = sum(ca[t] * cb[t] for t in common)
        na = math.sqrt(sum(v * v for v in ca.values()))
        nb = math.sqrt(sum(v * v for v in cb.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return max(0.0, min(1.0, float(dot / (na * nb))))

    @staticmethod
    def _market_factor(job: JobDescription) -> float:
        """
        Description: Determine market competition penalty (>=1.0).
        Layer: L4
        Input: JobDescription
        Output: float market_competition_factor
        """
        if job.market_competition_factor is not None:
            try:
                v = float(job.market_competition_factor)
                return max(1.0, v)
            except Exception:
                return 1.0

        # derive from applicants_count deterministically
        n = job.applicants_count or 0
        # penalty grows slowly: 1.0 -> ~1.5 at 100 applicants -> ~2.0 around 1000
        return float(max(1.0, 1.0 + (math.log10(1 + max(0, n)) / 2.0)))

    @staticmethod
    def _interview_chance_breakdown(orchestration_state: OrchestrationState, components: MatchComponents) -> InterviewChanceBreakdown:
        """
        Description: Apply weighted InterviewChanceScore formula:
                     (0.45*Skills + 0.35*Exp + 0.20*ATS) / MarketFactor.
        Layer: L4
        Input: OrchestrationState weights + MatchComponents
        Output: InterviewChanceBreakdown
        """
        weights = InterviewChanceWeights(
            w1_skill_overlap=float(orchestration_state.meta.get("w1_skill_overlap", 0.45)),
            w2_experience_alignment=float(orchestration_state.meta.get("w2_experience_alignment", 0.35)),
            w3_ats_score=float(orchestration_state.meta.get("w3_ats_score", 0.20)),
        )
        comps = InterviewChanceComponents(
            skill_overlap=float(components.skill_overlap),
            experience_alignment=float(components.experience_alignment),
            ats_score=float(components.ats_score),
            market_competition_factor=float(components.market_competition_factor),
        )
        return InterviewChanceBreakdown(weights=weights, components=comps)

    @staticmethod
    def _rationale(matched_req: List[str], missing_req: List[str], components: MatchComponents) -> List[str]:
        """
        Description: Generate compact, ATS-friendly rationale bullets for explainability.
        Layer: L4
        Input: matched/missing skills + components
        Output: list[str]
        """
        r = []
        r.append(f"Skill overlap: {components.skill_overlap:.2f} (matched {len(matched_req)} required skills).")
        if missing_req:
            r.append(f"Top gaps (required): {', '.join(missing_req[:8])}.")
        r.append(f"Experience alignment: {components.experience_alignment:.2f} (text similarity proxy).")
        r.append(f"ATS score: {components.ats_score:.2f} (structure/density proxy).")
        r.append(f"Market factor: {components.market_competition_factor:.2f} (competition penalty).")
        return r
