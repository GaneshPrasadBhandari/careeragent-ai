from careeragent.agents.drafting_agent_service import (
    _build_fallback_resume,
    _sanitize_resume_markdown,
)


def test_fallback_resume_includes_key_projects_for_senior_profile() -> None:
    profile = {
        "name": "Senior Candidate",
        "summary": "Architect with long-track delivery experience.",
        "skills": ["Python", "Fraud Detection"],
        "experience": [
            {
                "title": "Lead Data Scientist",
                "company": "Modefin",
                "start_date": "2012-01",
                "end_date": "2024-01",
                "bullets": ["Spearheaded fraud detection model development for BFSI customers."],
            }
        ],
        "education": [{"degree": "MS", "institution": "Clark", "graduation_year": "2026"}],
    }
    resume = _build_fallback_resume(profile, "Principal AI Architect", "Need fraud detection and BFSI ML")
    assert "## Key Projects" in resume


def test_resume_sanitizer_removes_star_headers() -> None:
    resume = """# Candidate\nmail@example.com\n\n## Summary\nSituation: owned migration\n\n## Skills\n- Python\n\n## Experience\nAction: Built platform\nResult: improved reliability\n\n## Education\n- MS"""
    out = _sanitize_resume_markdown(resume, profile={"skills": ["Python"]}, title="Engineer", jd="")
    assert "Situation:" not in out
    assert "Action:" not in out
    assert "Result:" not in out

