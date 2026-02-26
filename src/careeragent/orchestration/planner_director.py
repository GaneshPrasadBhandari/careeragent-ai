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


ATS_NOISE_TERMS = ["Shine", "Naukri", "TimesJobs", "Freshersworld", "Monster"]


def _resolve_location_terms(state: AgentState) -> List[str]:
    pref_loc = (state.preferences.location or "").strip()
    profile_loc = str((state.extracted_profile or {}).get("contact", {}).get("location") or "").strip()
    resolved = pref_loc or profile_loc or "United States, Remote"
    parts = [x.strip() for x in resolved.replace(";", ",").split(",") if x.strip()]
    terms = [resolved]
    terms.extend(parts[:2])
    if (state.preferences.country or "US").strip().upper() in {"US", "USA", "UNITED STATES"}:
        terms.extend(["United States", "USA", "Remote"])
    return list(dict.fromkeys([t for t in terms if t]))[:6]


def _default_negative_terms() -> List[str]:
    return ATS_NOISE_TERMS[:]


class Planner:
    """Description: Planner builds dynamic search personas from profile + preferences.
    Layer: L2
    Input: AgentState
    Output: state.search_personas
    """

    def build_personas(self, state: AgentState) -> List[PlannerPersona]:
        roles = state.preferences.target_roles or []
        primary = roles[0] if roles else "Solution Architect"

        must = [primary]
        if "architect" in primary.lower() or "solution" in primary.lower():
            must += ["Solution Architect", "AI Solution Architect", "GenAI Architect", "Solutions Architecture"]

        negative = _default_negative_terms()
        location_terms = _resolve_location_terms(state)
        geo_hint = location_terms[0] if location_terms else "United States, Remote"

        pA = PlannerPersona(
            persona_id="A",
            name=f"Strict ATS {geo_hint} 36h",
            strategy="ats_preferred",
            recency_hours=min(36.0, state.preferences.recency_hours),
            must_include=list(dict.fromkeys(must + location_terms[:2])),
            negative_terms=negative,
            site_filters=ATS_SITES,
        )
        pB = PlannerPersona(
            persona_id="B",
            name=f"ATS-preferred {geo_hint} 7d",
            strategy="ats_preferred",
            recency_hours=max(168.0, state.preferences.recency_hours),
            must_include=list(dict.fromkeys(must + ["remote", "hybrid"] + location_terms[:2])),
            negative_terms=negative,
            site_filters=ATS_SITES,
        )
        pC = PlannerPersona(
            persona_id="C",
            name=f"Broad {geo_hint} title-strict",
            strategy="broad",
            recency_hours=max(168.0, state.preferences.recency_hours),
            must_include=list(dict.fromkeys([primary] + location_terms[:2])),
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
        """Return (should_retry, action_reason).

        Description:
          Prevent endless refinement loops that hide the HITL approval gate.

        Layer: L3/L5
        Input: viable_count + batch_score
        Output: should_retry + reason

        Policy:
          - If ANY viable jobs exist, proceed to HITL.
          - Only retry/refine when we truly have *zero* viable jobs.

        Why:
          Phase2Evaluator already provides soft feedback when viable_count is low.
          Forcing retries in that case makes the UI look stuck at L5.
        """

        if viable_count >= 1:
            return False, "viable jobs exist; proceed to HITL with advisory"

        return True, "no viable jobs; relax constraints"

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
