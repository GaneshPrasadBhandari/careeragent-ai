import pytest
pytest.importorskip("fastapi")

import os

from careeragent.api.main import (
    _augment_scored_jobs,
    _build_cover_letter_text,
    _build_learning_resource_pack,
    _langsmith_status,
    _normalize_config,
    _phase6_qualified_jobs,
    _record_feedback_event,
)


def test_langsmith_status_uses_boolean_env(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_PROJECT", "careeragent-ai")
    status = _langsmith_status("run123")
    assert status["enabled"] is True
    assert "o/default/projects/p/careeragent-ai" in str(status["dashboard_url"])


def test_normalize_config_includes_new_limits():
    cfg = _normalize_config({})
    assert cfg["draft_jobs_limit"] == 0
    assert cfg["apply_jobs_limit"] == 0
    assert cfg["match_threshold"] == 0.40
    assert cfg["max_jobs"] == 140


def test_feedback_event_updates_learning_loop():
    state = {"learning_loop": {"user_feedback": 0, "employer_feedback": 0, "accepted": 0, "rejected": 0}}
    event = _record_feedback_event(
        state,
        {"source": "employer", "text": "We would like to schedule an interview next week.", "meta": {"company": "Acme"}},
    )
    assert event["evaluation"]["is_genuine"] is True
    assert state["learning_loop"]["employer_feedback"] == 1
    assert state["employer_outcomes"]["interview"] == 1


def test_cover_letter_format_is_classic():
    profile = {"name": "Alex", "email": "alex@example.com", "phone": "123", "skills": ["Python"]}
    job = {"title": "AI Engineer", "company": "ExampleCo"}
    cover = _build_cover_letter_text(profile, job)
    assert cover.splitlines()[0] == "Alex"
    assert "Subject: Application for AI Engineer" in cover
    assert "Sincerely," in cover


def test_scored_jobs_include_rationale():
    jobs = [{"title": "ML Eng", "description": "python ml", "score": 0.8, "jd_alignment_percent": 75, "posted_hours_ago": 6, "remote": True}]
    profile = {"skills": ["Python", "ML"]}
    out = _augment_scored_jobs(jobs, profile)
    assert out[0]["recommendation_rationale"]
    assert any("Context fit" in line for line in out[0]["recommendation_rationale"])
    assert out[0]["match_explanation"]
    assert "Resume Skills vs. Job Requirements" in out[0]["skill_comparison_prompt"]


def test_normalize_config_handles_malformed_nested_values():
    cfg = _normalize_config({"notifications": "bad", "work_modes": "remote", "geo_preferences": []})
    assert isinstance(cfg["notifications"], dict)
    assert cfg["notifications"]["email"] == ""
    assert cfg["work_modes"] == ["remote", "hybrid", "onsite"]
    assert cfg["geo_preferences"] == {"remote": True, "locations": [], "country_selector": "US"}


def test_phase6_qualification_keeps_most_jobs_above_half_score():
    scored = [
        {"id": f"job_{idx}", "score": score, "interview_probability_percent": score * 100}
        for idx, score in enumerate([0.83, 0.79, 0.74, 0.71, 0.68, 0.65, 0.61, 0.57, 0.54, 0.49], start=1)
    ]
    out = _phase6_qualified_jobs(scored, 0.72)
    kept = {job["id"] for job in out}
    above_half = [job for job in scored if job["score"] > 0.5]
    retained = [job for job in above_half if job["id"] in kept]
    assert len(retained) >= 7


def test_feedback_event_creates_self_learning_prompt():
    state = {"learning_loop": {"user_feedback": 0, "employer_feedback": 0, "accepted": 0, "rejected": 0}}
    _record_feedback_event(state, {"source": "user", "rating": 4, "text": "Ranking was too strict for adjacent AI roles."})
    assert "Self-Learning Optimization Prompt" in state["self_learning_prompt"]


def test_learning_resource_pack_contains_direct_links():
    pack = _build_learning_resource_pack("LangChain")
    assert "http" in pack["official_documentation"]
    assert "youtube.com" in pack["youtube_search"]
    assert len(pack["top_websites"]) == 3
