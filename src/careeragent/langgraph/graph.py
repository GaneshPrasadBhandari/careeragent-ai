from __future__ import annotations

from typing import Any, Dict

from careeragent.langgraph.state import CareerGraphState
from careeragent.langgraph.nodes import (
    l0_security_node,
    l2_parser_node,
    l3_discovery_node,
    l4_match_node,
    l5_rank_node,
    evaluator_node,
)


def build_graph():
    """
    Description: Build LangGraph orchestration with evaluator after each layer.
    Layer: L6
    Input: CareerGraphState
    Output: Compiled graph
    """
    from langgraph.graph import StateGraph, END  # local import to avoid import errors if not installed

    g = StateGraph(CareerGraphState)

    g.add_node("L0_SECURITY", l0_security_node)
    g.add_node("L2_PARSE", l2_parser_node)
    g.add_node("L3_DISCOVERY", l3_discovery_node)
    g.add_node("L4_MATCH", l4_match_node)
    g.add_node("L5_RANK", l5_rank_node)

    # Evaluator wrappers (we pass score computed in-state; in production you compute score per layer)
    async def eval_after_parse(state: CareerGraphState):
        prof = state.get("profile") or {}
        skills = len(prof.get("skills") or [])
        score = min(1.0, 0.35 + (skills / 30.0))
        fb = [] if score >= 0.7 else ["Profile thin: add skills + bullets."]
        return await evaluator_node(state, "L2", "parser", score, fb)

    async def eval_after_discovery(state: CareerGraphState):
        n = len(state.get("jobs_raw") or [])
        score = 0.2 if n < 8 else (0.6 if n < 20 else 0.8)
        fb = [] if score >= 0.7 else ["Low discovery volume: refine query or widen filters."]
        return await evaluator_node(state, "L3", "discovery", score, fb)

    async def eval_after_match(state: CareerGraphState):
        jobs = state.get("jobs_scored") or []
        top = float(jobs[0]["score"]) if jobs else 0.0
        score = top
        fb = [] if score >= 0.7 else ["Top match low: broaden roles or improve resume keywords."]
        return await evaluator_node(state, "L4", "match", score, fb)

    g.add_node("EVAL_L2", eval_after_parse)
    g.add_node("EVAL_L3", eval_after_discovery)
    g.add_node("EVAL_L4", eval_after_match)

    g.set_entry_point("L0_SECURITY")

    # Flow
    g.add_edge("L0_SECURITY", "L2_PARSE")
    g.add_edge("L2_PARSE", "EVAL_L2")

    # Conditional: evaluator may set HITL
    def route_from_eval(state: CareerGraphState, layer_key: str):
        if state.get("status") == "needs_human_approval" or state.get("status") == "blocked":
            return "END"
        # retry logic: if last gate decision was retry, loop back to same layer
        gates = state.get("gates") or []
        if gates and getattr(gates[-1], "decision", None) == "retry":  # GateEvent dataclass
            return layer_key
        return "NEXT"

    g.add_conditional_edges("EVAL_L2", lambda s: route_from_eval(s, "L2_PARSE"), {"L2_PARSE": "L2_PARSE", "NEXT": "L3_DISCOVERY", "END": END})
    g.add_edge("L3_DISCOVERY", "EVAL_L3")
    g.add_conditional_edges("EVAL_L3", lambda s: route_from_eval(s, "L3_DISCOVERY"), {"L3_DISCOVERY": "L3_DISCOVERY", "NEXT": "L4_MATCH", "END": END})
    g.add_edge("L4_MATCH", "EVAL_L4")
    g.add_conditional_edges("EVAL_L4", lambda s: route_from_eval(s, "L4_MATCH"), {"L4_MATCH": "L4_MATCH", "NEXT": "L5_RANK", "END": END})

    # Rank ends in HITL (approval) in your product; here we stop after ranking
    g.add_edge("L5_RANK", END)

    return g.compile()