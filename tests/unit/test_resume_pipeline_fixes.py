from careeragent.agents.drafting_agent_service import _sanitize_cover_letter, _sanitize_resume_markdown
from careeragent.agents.leadscout_service import JobLead, LeadScoutService
from careeragent.core.state import AgentState, Preferences
from careeragent.orchestration.planner_director import Planner


def test_planner_respects_user_recency_window() -> None:
    state = AgentState(run_id="r-recent", preferences=Preferences(recency_hours=24.0, target_roles=["AI Engineer"]))
    state.extracted_profile = {"skills": ["Azure", "LLM"]}
    personas = Planner().build_personas(state)
    by_id = {p.persona_id: p for p in personas}
    assert by_id["A"].recency_hours <= 36.0
    assert by_id["B"].recency_hours <= 24.0
    assert by_id["C"].recency_hours <= 24.0


def test_leadscout_filters_closed_and_old_jobs() -> None:
    leads = [
        JobLead(id="1", title="AI Engineer", company="A", url="https://x/1", posted_date="1 hour ago"),
        JobLead(id="2", title="ML Engineer", company="B", url="https://x/2", posted_date="4 days ago"),
        JobLead(id="3", title="Data", company="C", url="https://x/3", description="application closed"),
    ]
    cleaned = LeadScoutService._filter_unavailable_jobs(leads)
    assert len(cleaned) == 2
    recent = LeadScoutService._filter_by_recency(cleaned, recency_hours=24)
    assert len(recent) == 1
    assert recent[0].id == "1"


def test_sanitize_resume_and_cover_outputs() -> None:
    profile = {"name": "Ganesh Prasad Bhandari", "skills": ["Azure", "Python"], "summary": "AI architect"}
    bad_resume = """# Ganesh Prasad Bhandari
Professional Summary
508-365-9302 | gbhandari@clarku.com
508-365-9302 | gbhandari@clarku.com
## Professional Summary
Experienced.
## Technical Skills
- **AI/ML:** azure ml
## Experience
- Did things
## Education
- MS
"""
    fixed = _sanitize_resume_markdown(bad_resume, profile=profile, title="AI Engineer", jd="Azure OpenAI")
    assert "## Summary" in fixed
    assert "## Skills" in fixed
    assert fixed.lower().count("gbhandari@clarku.com") <= 1

    bad_cover = "Cover Letter\nI am excited for this role."
    fixed_cover = _sanitize_cover_letter(bad_cover, profile=profile, title="AI Engineer")
    assert fixed_cover.startswith("Dear Hiring Manager,")
    assert "Sincerely" in fixed_cover
