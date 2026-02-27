from careeragent.agents.auto_applier_agent_service import AutoApplierAgentService
from careeragent.agents.drafting_agent_service import optimize_resume_keywords
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, Preferences
from careeragent.orchestration.planner_director import Planner


def test_planner_builds_three_cognitive_clusters() -> None:
    state = AgentState(run_id="r1", preferences=Preferences(target_roles=["Solutions Architect"]))
    state.extracted_profile = {"skills": ["Enterprise Architecture", "Agentic AI"]}
    personas = Planner().build_personas(state)
    assert len(personas) == 3
    assert any("Enterprise Architecture" in p.must_include for p in personas if p.persona_id == "A")
    assert any("LLM" in p.must_include for p in personas if p.persona_id == "B")
    assert any("Technical Lead" in p.must_include for p in personas if p.persona_id == "C")


def test_resume_keyword_optimizer_injects_missing_terms() -> None:
    resume = "# ATS Resume\n## Summary\nExperienced architect."
    jd = "Need Python, Machine Learning, LLM, MLOps and Enterprise Architecture experience."
    optimized, injected = optimize_resume_keywords(resume_md=resume, jd_text=jd, profile_skills=["Python"])
    assert "## Skills Highlights" in optimized
    assert len(injected) >= 3


def test_auto_applier_requires_hitl_for_mid_scores() -> None:
    settings = Settings()
    service = AutoApplierAgentService(settings)
    state = AgentState(run_id="r2", preferences=Preferences())
    state.approved_job_urls = ["https://example.com/job/1"]
    state.ranking = [{"url": "https://example.com/job/1", "phase2_score": 0.72}]
    out = service.apply(state, dry_run=True)
    assert out
    assert out[0].status == "hitl_required"
