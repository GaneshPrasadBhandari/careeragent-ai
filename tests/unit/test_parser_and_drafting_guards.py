from careeragent.agents.drafting_agent_service import _strip_markdown
from careeragent.agents.parser_agent_service import _validate_and_backfill_skills


def test_skill_validator_backfills_from_skill_sections_without_hardcoded_vocab() -> None:
    text = """
    Core Competencies: Stakeholder Communication, Team Leadership, Customer Discovery
    Skills: Python, SQL, Airflow
    Experience
    """
    out = _validate_and_backfill_skills(text, current_skills=[])
    lowered = {s.lower() for s in out}
    assert {"python", "sql", "airflow", "stakeholder communication", "team leadership"}.issubset(lowered)


def test_markdown_stripper_removes_common_tokens() -> None:
    md = "# Header\n- **Bold Point**\n1. `code`\n[Link](https://example.com)"
    plain = _strip_markdown(md)
    assert "#" not in plain
    assert "**" not in plain
    assert "`" not in plain
    assert "[" not in plain
    assert "Header" in plain
    assert "Bold Point" in plain
