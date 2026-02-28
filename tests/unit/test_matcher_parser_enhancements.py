from careeragent.agents.matcher_agent_service import MatcherAgentService
from careeragent.agents.parser_agent_service import _infer_latent_skills
from careeragent.core.state import AgentState, Preferences


def test_infer_latent_skills_from_resume_context() -> None:
    text = "Built LLM RAG systems on AWS and partnered with cross-functional stakeholders"
    inferred = _infer_latent_skills(text, existing=["Python"])
    low = {x.lower() for x in inferred}
    assert "ai/ml" in low
    assert "cloud engineering" in low
    assert "stakeholder management" in low


def test_matcher_emits_probability_fields_for_dashboard_contract() -> None:
    state = AgentState(run_id="r-m", preferences=Preferences())
    state.extracted_profile = {
        "skills": ["Python", "AWS", "Machine Learning", "LangGraph"],
        "experience": [{"title": "Senior AI Architect", "start_date": "2018", "end_date": "2024"}],
    }
    state.jobs_raw = [
        {
            "title": "ML Architect",
            "snippet": "Need Python, AWS, Machine Learning, LLM and LangGraph experience",
            "url": "https://example.com/job",
        }
    ]
    out = MatcherAgentService().score_jobs(state)
    assert out and out[0]["interview_probability_percent"] > 0
    assert "job_offer_probability_percent" in out[0]
    assert "ats_keyword_match_percent" in out[0]
