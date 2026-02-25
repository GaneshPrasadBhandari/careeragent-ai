from pathlib import Path

def test_imports_do_not_crash():
    import sys
    sys.path.insert(0, "src")

    # Modules touched by the patch must import cleanly.
    from careeragent.orchestration.orchestrator import Orchestrator  # noqa: F401
    from careeragent.orchestration.planner_director import Director  # noqa: F401
    from careeragent.agents.matcher_agent_service import MatcherAgentService  # noqa: F401
    from careeragent.agents.drafting_agent_service import DraftingAgentService  # noqa: F401
    from careeragent.core.mcp_client import MCPClient  # noqa: F401
    from careeragent.core.state_store import StateStore  # noqa: F401

def test_env_example_contains_no_real_keys():
    # Prevent accidental secrets from being committed again
    txt = Path(".env_example").read_text(encoding="utf-8", errors="ignore")
    assert "tvly-dev-" not in txt, "Remove real Tavily keys from .env_example"
    assert "sk-" not in txt, "Looks like an API key leaked into .env_example"
