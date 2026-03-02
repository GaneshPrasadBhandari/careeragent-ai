from careeragent.agents.parser_agent_service import ParserAgentService
from careeragent.core.settings import Settings
from careeragent.managers.managers import ExtractionManager


def test_parser_handles_escaped_newlines_and_key_skills_sections() -> None:
    raw_resume = (
        "Ganesh Prasad Bhandari\\n"
        "Key Skills\\n"
        "Programming and Tools: Python, TensorFlow, Scikit-learn, SQL\\n"
        "Cloud Platforms: Azure OpenAI, AWS, Azure ML Studio, LLM Ops\\n"
        "Notable Projects\\n"
        "AI-Powered Copilot & AI Assistants - NTT DATA\\n"
    )
    svc = ParserAgentService(Settings())
    parsed, normalized = svc.parse_from_upload(filename="resume.txt", file_bytes=b"", raw_text=raw_resume)

    assert "\n" in normalized
    low_skills = {s.lower() for s in parsed.skills}
    assert "python" in low_skills
    assert "azure openai" in low_skills or "azure" in low_skills
    assert any("copilot" in p.lower() for p in parsed.projects)


def test_extraction_manager_scores_architect_roles_from_full_text_and_synonyms() -> None:
    manager = ExtractionManager()
    profile = {
        "skills": ["Generative AI", "Azure OpenAI", "Solution Architecture", "Python"],
        "experience": [{"title": "Senior Solution Architect", "years": 8}],
    }
    leads = [
        {
            "title": "GenAI Solution Architect",
            "full_text_md": "Looking for AI solution architect with Azure OpenAI and Python for enterprise delivery.",
            "snippet": "GenAI architecture role",
            "company": "ExampleCo",
        }
    ]

    scored = manager.extract_and_score(leads, profile, threshold=0.45)
    assert scored
    assert scored[0]["score"] >= 0.55
    assert "python" in scored[0].get("matched_skills", [])
