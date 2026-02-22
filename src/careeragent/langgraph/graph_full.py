
from __future__ import annotations

from careeragent.langgraph.state import CareerGraphState
from careeragent.langgraph.nodes import (
    l0_security_node,
    l2_parser_node,
    l3_discovery_node,
    l4_match_node,
    l5_rank_node,
)
from careeragent.langgraph.nodes_l6_l9 import (
    l6_draft_node, l6_evaluator_node,
    l7_apply_node, l7_evaluator_node,
    l8_tracker_node, l8_evaluator_node,
    l9_analytics_node,
)

def build_hunt_graph():
    """
    Description: Build L0->L5 hunt graph (stops at ranking/HITL).
    Layer: L6
    """
    from langgraph.graph import StateGraph, END

    g = StateGraph(CareerGraphState)
    g.add_node("L0_SECURITY", l0_security_node)
    g.add_node("L2_PARSE", l2_parser_node)
    g.add_node("L3_DISCOVERY", l3_discovery_node)
    g.add_node("L4_MATCH", l4_match_node)
    g.add_node("L5_RANK", l5_rank_node)

    g.set_entry_point("L0_SECURITY")
    g.add_edge("L0_SECURITY", "L2_PARSE")
    g.add_edge("L2_PARSE", "L3_DISCOVERY")
    g.add_edge("L3_DISCOVERY", "L4_MATCH")
    g.add_edge("L4_MATCH", "L5_RANK")
    g.add_edge("L5_RANK", END)

    return g.compile()


def build_finalize_graph():
    """
    Description: Build L6->L9 finalize graph (draft->apply->track->analytics).
    Layer: L6
    """
    from langgraph.graph import StateGraph, END

    g = StateGraph(CareerGraphState)
    g.add_node("L6_DRAFT", l6_draft_node)
    g.add_node("EVAL_L6", l6_evaluator_node)
    g.add_node("L7_APPLY", l7_apply_node)
    g.add_node("EVAL_L7", l7_evaluator_node)
    g.add_node("L8_TRACK", l8_tracker_node)
    g.add_node("EVAL_L8", l8_evaluator_node)
    g.add_node("L9_ANALYTICS", l9_analytics_node)

    g.set_entry_point("L6_DRAFT")
    g.add_edge("L6_DRAFT", "EVAL_L6")
    g.add_edge("EVAL_L6", "L7_APPLY")
    g.add_edge("L7_APPLY", "EVAL_L7")
    g.add_edge("EVAL_L7", "L8_TRACK")
    g.add_edge("L8_TRACK", "EVAL_L8")
    g.add_edge("EVAL_L8", "L9_ANALYTICS")
    g.add_edge("L9_ANALYTICS", END)

    return g.compile()
