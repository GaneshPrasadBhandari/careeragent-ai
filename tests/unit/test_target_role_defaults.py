from careeragent.api import main as api_main


def test_normalize_config_defaults_to_ai_focused_roles():
    cfg = api_main._normalize_config({})
    assert cfg["target_roles"][:4] == [
        "AI Engineer",
        "AI/ML Solution Architect",
        "GenAI Solution Architect",
        "Principal Data Scientist",
    ]


def test_infer_target_roles_uses_ai_backbone_when_roles_are_generic():
    roles = api_main._infer_target_roles({"skills": []}, ["Software Engineer"])
    assert "AI Engineer" in roles
    assert "AI/ML Solution Architect" in roles
    assert "GenAI Solution Architect" in roles
    assert "Principal Data Scientist" in roles
