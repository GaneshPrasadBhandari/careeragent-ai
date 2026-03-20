import pytest
pytest.importorskip("fastapi")

import math
import os

from careeragent.api.approval_utils import qualified_from_state

from careeragent.api.main import (
    _augment_scored_jobs,
    _build_cover_letter_text,
    _build_intent,
    _build_learning_resource_pack,
    _extract_profile_from_text,
    _langsmith_status,
    _normalize_config,
    _phase6_qualified_jobs,
    _record_feedback_event,
    _reset_downstream_state,
    _sync_feedback_to_agent_brain,
    _stub_leads,
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


def test_phase6_qualification_is_selective_and_keeps_strong_diverse_jobs():
    scored = [
        {
            "id": f"job_{idx}",
            "score": score,
            "keyword_score": max(0.35, score - 0.08),
            "semantic_score": max(0.40, score - 0.05),
            "cognitive_score": score,
            "interview_probability_percent": score * 100,
            "cognitive_decision": {"approved": score >= 0.57},
            "source": "linkedin" if idx < 4 else ("indeed" if idx < 7 else "glassdoor"),
            "title": f"AI Engineer {idx}",
            "company": f"Company {idx}",
            "location": "Remote",
            "url": f"https://example.com/job/{idx}",
        }
        for idx, score in enumerate([0.83, 0.79, 0.74, 0.71, 0.68, 0.65, 0.61, 0.57, 0.54, 0.49], start=1)
    ]
    out = _phase6_qualified_jobs(scored, 0.72)
    kept = {job["id"] for job in out}
    assert len(out) >= math.ceil(len(scored) * 0.85)
    assert {"job_1", "job_2", "job_3"}.issubset(kept)
    assert len({job["url"] for job in out}) == len(out)


def test_feedback_event_creates_self_learning_prompt():
    state = {"learning_loop": {"user_feedback": 0, "employer_feedback": 0, "accepted": 0, "rejected": 0}}
    _record_feedback_event(state, {"source": "user", "rating": 4, "text": "Ranking was too strict for adjacent AI roles."})
    assert "Self-Learning Optimization Prompt" in state["self_learning_prompt"]


def test_feedback_event_accepts_job_payload_shape():
    state = {"learning_loop": {"user_feedback": 0, "employer_feedback": 0, "accepted": 0, "rejected": 0}}
    event = _record_feedback_event(
        state,
        {"source": "user", "job_id": "job-123", "rating": -1, "comment": "Too many US jobs for India"},
    )
    assert event["meta"]["job_id"] == "job-123"
    assert event["text"] == "Too many US jobs for India"


def test_learning_resource_pack_contains_direct_links():
    pack = _build_learning_resource_pack("LangChain")
    assert "http" in pack["official_documentation"]
    assert "youtube.com" in pack["youtube_search"]
    assert len(pack["top_websites"]) == 3


def test_resume_parser_recovers_senior_ai_titles():
    profile = _extract_profile_from_text("""Ganesh Prasad Bhandari
Professional Summary
Dynamic AI/ML professional with 16+ years of experience.
Professional Experience
Senior Solution Architect
Data Science Team Lead (Python)
Senior Data Scientist
Key Skills
Python, TensorFlow, Azure OpenAI, AWS
""")
    titles = [item["title"] for item in profile["experience"]]
    assert "Senior Solution Architect" in titles
    assert "Data Science Team Lead (Python)" in titles
    assert any(item["years"] >= 16 for item in profile["experience"])


def test_build_intent_prefers_resume_roles_over_generic_default():
    profile = {
        "skills": ["Python", "TensorFlow", "Azure OpenAI", "AWS", "Machine Learning"],
        "experience": [{"title": "Senior Solution Architect", "years": 16}],
        "raw_text": "16+ years AI ML Generative AI Azure OpenAI",
    }
    cfg = _normalize_config({})
    intent = _build_intent(profile, cfg)
    assert intent["target_roles"][0] == "Senior Solution Architect"
    assert "AI Solution Architect" in intent["target_roles"]


def test_build_intent_passes_self_learning_context():
    profile = {"skills": ["Python"], "experience": [], "raw_text": ""}
    cfg = _normalize_config({"self_learning_context": "Prefer India-first remote roles and broaden semantic matches."})
    intent = _build_intent(profile, cfg)
    assert "India-first remote roles" in intent["self_learning_context"]


def test_sync_feedback_to_agent_brain_creates_context():
    state = {
        "feedback_events": [{"source": "user", "rating": -1, "text": "Too many US jobs", "meta": {"job_id": "job-1"}}],
        "system_prompt_update": "broaden semantics",
        "learning_loop": {"user_feedback": 1, "employer_feedback": 0, "accepted": 1, "rejected": 0},
        "feedback_learning_state": {"strictness_mode": "less_strict", "targeting_mode": "broad_semantic"},
        "employer_outcomes": {},
        "apply_results": [],
        "interviews": [],
        "followup_queue": [],
    }
    context = _sync_feedback_to_agent_brain(state)
    assert context
    assert state["self_learning_context"] == context


def test_stub_leads_expand_to_unique_urls_and_phase6_dedupes_down_to_strong_jobs():
    profile = {
        "skills": ["Python", "TensorFlow", "Azure OpenAI", "AWS", "Machine Learning", "AI Architect"],
        "experience": [{"title": "Senior Solution Architect", "years": 16}],
    }
    leads = _stub_leads(profile, max_jobs=80)
    assert len({job["url"] for job in leads}) == 80

    scored = [
        {
            **job,
            "score": 0.62 if idx < 72 else 0.41,
            "cognitive_score": 0.95 if idx < 72 else 0.55,
            "interview_probability_percent": 78 if idx < 50 else 61,
            "cognitive_decision": {"approved": idx < 72},
        }
        for idx, job in enumerate(leads)
    ]
    approved = _phase6_qualified_jobs(scored, 0.40)
    assert len(approved) >= math.ceil(len(scored) * 0.85)
    assert len({job["url"] for job in approved}) == len(approved)


def test_stub_leads_use_search_urls_for_us_and_skip_naukri():
    profile = {"skills": ["Python"], "experience": [{"title": "AI Engineer", "years": 10}]}
    leads = _stub_leads(
        profile,
        max_jobs=16,
        config={"geo_preferences": {"country_selector": "US", "locations": ["Boston, MA"]}},
    )
    assert leads
    assert all("naukri.com" not in str(job["url"]).lower() for job in leads)
    assert {job["source"] for job in leads[:8]} == {"linkedin", "indeed", "glassdoor", "ziprecruiter", "google_jobs", "greenhouse", "lever", "workday"}
    assert any("/jobs/search" in str(job["url"]) for job in leads if job["source"] == "linkedin")
    assert any("indeed.com/jobs" in str(job["url"]) for job in leads if job["source"] == "indeed")
    assert any("glassdoor.com/Job/jobs.htm" in str(job["url"]) for job in leads if job["source"] == "glassdoor")
    assert all("/view/" not in str(job["url"]) for job in leads if job["source"] in {"linkedin", "indeed", "glassdoor", "ziprecruiter", "google_jobs", "greenhouse", "lever", "workday"})


def test_stub_leads_allow_naukri_for_india():
    profile = {"skills": ["Python"], "experience": [{"title": "AI Engineer", "years": 10}]}
    leads = _stub_leads(
        profile,
        max_jobs=14,
        config={"geo_preferences": {"country_selector": "IN", "locations": ["Bengaluru, India"]}},
    )
    assert any("naukri.com" in str(job["url"]).lower() for job in leads)


def test_qualified_from_state_requires_explicit_approval():
    state = {"layer_debug": {"L5": {"qualified_jobs": [{"id": "job-1"}]}}}
    assert qualified_from_state(state) == []
    state["approved_jobs"] = [{"id": "job-1"}]
    assert qualified_from_state(state) == [{"id": "job-1"}]


def test_feedback_event_persists_required_identity_fields(tmp_path, monkeypatch):
    from careeragent.api import main as api_main

    ledger = tmp_path / "feedback_ledger.jsonl"
    monkeypatch.setattr(api_main, "FEEDBACK_LEDGER_FILE", ledger)
    state = {
        "run_id": "run-123",
        "profile": {"email": "senior@example.com", "experience": [{"title": "Senior Solution Architect", "years": 16}]},
        "learning_loop": {"user_feedback": 0, "employer_feedback": 0, "accepted": 0, "rejected": 0},
    }
    event = _record_feedback_event(state, {"source": "user", "rating": 5, "text": "Great architect-level matches."})
    assert event["timestamp"]
    assert event["user_email"] == "senior@example.com"
    assert event["user_role"] == "Senior Solution Architect"
    assert event["run_id"] == "run-123"
    assert event["feedback_text"] == "Great architect-level matches."


def test_sync_feedback_to_agent_brain_creates_system_instruction_update(tmp_path, monkeypatch):
    from careeragent.api import main as api_main

    ledger = tmp_path / "feedback_ledger.jsonl"
    monkeypatch.setattr(api_main, "FEEDBACK_LEDGER_FILE", ledger)
    state = {
        "run_id": "run-123",
        "feedback_events": [{
            "timestamp": "2026-03-19T12:00:00+00:00",
            "source": "user",
            "rating": 5,
            "text": "Prefer senior architect roles only",
            "feedback_text": "Prefer senior architect roles only",
            "user_role": "Senior Solution Architect",
            "user_email": "senior@example.com",
            "run_id": "run-123",
            "meta": {},
        }],
        "learning_loop": {"user_feedback": 1, "employer_feedback": 0, "accepted": 1, "rejected": 0},
        "feedback_learning_state": {"strictness_mode": "balanced", "targeting_mode": "more_targeted"},
        "employer_outcomes": {},
        "apply_results": [],
        "interviews": [],
        "followup_queue": [],
    }
    _sync_feedback_to_agent_brain(state)
    assert "ranking_reasoner" in state["system_instruction_update"]
    assert "evaluator_guardrails" in state["system_instruction_update"]
