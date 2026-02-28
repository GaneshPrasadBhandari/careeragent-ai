from careeragent.agents.parser_agent_service import ExtractedResume, ParserEvaluatorL2, EducationModel


def test_parser_evaluator_merges_when_phase2_has_more_skills_and_education() -> None:
    p1 = ExtractedResume(skills=["Python", "AWS"], education=[])
    p2 = ExtractedResume(
        skills=["Python", "AWS", "LLM", "RAG"],
        education=[EducationModel(institution="MS Computer Science")],
        projects=["Built GenAI assistant"],
    )
    merged, score, low_conf = ParserEvaluatorL2(retry_limit=2, threshold=0.85).evaluate(p1, p2)
    assert len(merged.skills) >= 4
    assert merged.education
    assert score > 0
    assert low_conf in {True, False}
