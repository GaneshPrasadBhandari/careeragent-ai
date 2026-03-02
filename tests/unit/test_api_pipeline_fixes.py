import pytest
pytest.importorskip("fastapi")

import os

from careeragent.api.main import (
    _augment_scored_jobs,
    _build_cover_letter_text,
    _langsmith_status,
    _normalize_config,
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
