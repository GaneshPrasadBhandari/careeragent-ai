import os

from careeragent.api.main import _build_resume_markdown
from careeragent.core.settings import Settings, bootstrap_langsmith


def test_resume_markdown_strips_sta_labels_and_urls_for_senior_profiles() -> None:
    profile = {
        "name": "Test Candidate",
        "email": "test@example.com",
        "phone": "+14155550100",
        "summary": "Principal architect profile https://example.com/portfolio",
        "skills": ["Python", "LLM", "Azure"],
        "experience": [{"title": "Principal AI Architect", "years": 12}],
        "projects": ["Program A", "Program B", "Program C", "Program D"],
        "education": ["MS Computer Science"],
    }
    resume_md = _build_resume_markdown(profile, keyword_hints=["llm", "azure"], job={"title": "Principal AI Architect"})
    assert "Situation:" not in resume_md
    assert "Task:" not in resume_md
    assert "Action:" not in resume_md
    assert "http://" not in resume_md and "https://" not in resume_md
    assert resume_md.count("### Project") >= 4


def test_bootstrap_defaults_to_beta_project(monkeypatch) -> None:
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    s = Settings(LANGCHAIN_TRACING_V2="true")
    bootstrap_langsmith(s)
    assert os.environ["LANGCHAIN_PROJECT"] == "careeragent-ai-beta"
    assert os.environ["LANGSMITH_PROJECT"] == "careeragent-ai-beta"
