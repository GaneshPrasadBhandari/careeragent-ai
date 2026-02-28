"""Compatibility wrapper for parser agent.
Implements dual-phase cognitive extraction through ParserAgentService.
"""

from careeragent.agents.parser_agent_service import (
    ParserAgentService,
    ExtractedResume,
    ParserEvaluatorL2,
)

__all__ = ["ParserAgentService", "ExtractedResume", "ParserEvaluatorL2"]
