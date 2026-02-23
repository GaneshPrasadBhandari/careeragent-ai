from __future__ import annotations

from typing import List, Tuple

from careeragent.core.state import AgentState, PlannerPersona


ATS_SITES = [
    "greenhouse.io",
    "lever.co",
    "workdayjobs.com",
    "icims.com",
    "myworkdayjobs.com",
    "successfactors.com",
    "smartrecruiters.com",
]


class Planner:
    """Description: Planner builds dynamic search personas from profile + preferences.
    Layer: L2
    Input: AgentState
    Output: state.search_personas
    """

    def build_personas(self, state: AgentState) -> List[PlannerPersona]:
        roles = state.preferences.target_roles or []
        # infer a primary role phrase
        primary = roles[0] if roles else "Solution Architect"

        must = [primary]
        # add architecture synonyms for SA roles
        if "architect" in primary.lower() or "solution" in primary.lower():
            must += ["Solution Architect", "AI Solution Architect", "GenAI Architect", "Solutions Architecture"]

        negative = ["India", "Bangalore", "Nashik", "Shine", "Naukri", "TimesJobs"]

        pA = PlannerPersona(
            persona_id="A",
            name="Strict ATS US 36h",
            strategy="ats_preferred",
            recency_hours=min(36.0, state.preferences.recency_hours),
            must_include=must,
            negative_terms=negative,
            site_filters=ATS_SITES,
        )
        pB = PlannerPersona(
            persona_id="B",
            name="ATS-preferred Remote US 7d",
            strategy="ats_preferred",
            recency_hours=max(168.0, state.preferences.recency_hours),
            must_include=must + ["remote", "hybrid"],
            negative_terms=negative,
            site_filters=ATS_SITES,
        )
        pC = PlannerPersona(
            persona_id="C",
            name="Broad US title-strict",
            strategy="broad",
            recency_hours=max(168.0, state.preferences.recency_hours),
            must_include=[primary, "United States"],
            negative_terms=negative,
            site_filters=[],
        )

        return [pA, pB, pC]


class Director:
    """Description: Director enforces soft-fencing and prevents wipeout.
    Layer: L2
    Input: AgentState + evaluation outcomes
    Output: strategy shifts / relax constraints / stop
    """

    def soft_fence(self, state: AgentState, *, viable_count: int, batch_score: float) -> Tuple[bool, str]:
        """Return (should_retry, action_reason)."""

        # If we have any viable jobs, NEVER wipe out to 0; proceed to HITL.
        if viable_count >= 3:
            return False, "enough viable jobs"

        # If some jobs exist but low, try persona shift before retrying same plan.
        if viable_count > 0:
            return True, "low viable count; shift persona / relax"

        # 0 results: retry, but with relaxed constraints
        return True, "no results; relax constraints"

    def next_persona(self, state: AgentState) -> str:
        personas = [p.persona_id for p in state.search_personas]
        cur = state.active_persona_id or (personas[0] if personas else "A")
        if cur in personas:
            idx = personas.index(cur)
            nxt = personas[min(idx + 1, len(personas) - 1)]
            return nxt
        return personas[0] if personas else "A"

    def relax_constraints(self, state: AgentState) -> None:
        # widen recency and soften strategy
        state.preferences.recency_hours = min(168.0, max(state.preferences.recency_hours, 72.0))
        state.query_modifiers["strategy"] = "ats_preferred"
