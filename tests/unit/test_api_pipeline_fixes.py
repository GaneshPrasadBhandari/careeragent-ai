import importlib
import sys
import types


def _import_api_main_with_stubs():
    try:
        return importlib.import_module("careeragent.api.main")
    except Exception:
        sys.modules.pop("careeragent.api.main", None)
        fastapi = types.ModuleType("fastapi")

        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return None

        class _FastAPI(_Dummy):
            def add_middleware(self, *args, **kwargs):
                return None

            def get(self, *args, **kwargs):
                def _decorator(fn):
                    return fn
                return _decorator

            post = get

            def on_event(self, *args, **kwargs):
                def _decorator(fn):
                    return fn
                return _decorator

        fastapi.BackgroundTasks = _Dummy
        fastapi.FastAPI = _FastAPI
        fastapi.File = lambda *args, **kwargs: None
        fastapi.Form = lambda *args, **kwargs: None

        class _HTTPException(Exception):
            def __init__(self, status_code: int = 500, detail: str = ""):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        fastapi.HTTPException = _HTTPException
        fastapi.UploadFile = _Dummy

        middleware = types.ModuleType("fastapi.middleware")
        cors = types.ModuleType("fastapi.middleware.cors")
        cors.CORSMiddleware = _Dummy

        responses = types.ModuleType("fastapi.responses")
        responses.FileResponse = _Dummy
        responses.JSONResponse = _Dummy

        sys.modules.setdefault("fastapi", fastapi)
        sys.modules.setdefault("fastapi.middleware", middleware)
        sys.modules.setdefault("fastapi.middleware.cors", cors)
        sys.modules.setdefault("fastapi.responses", responses)

        return importlib.import_module("careeragent.api.main")


api_main = _import_api_main_with_stubs()

import os

_augment_scored_jobs = api_main._augment_scored_jobs
_build_cover_letter_text = api_main._build_cover_letter_text
_langsmith_status = api_main._langsmith_status
_normalize_config = api_main._normalize_config
_stub_leads = api_main._stub_leads
_record_feedback_event = api_main._record_feedback_event
_is_duplicate_action = api_main._is_duplicate_action
_mark_action_processed = api_main._mark_action_processed


def test_langsmith_status_uses_boolean_env(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_PROJECT", "careeragent-ai")
    status = _langsmith_status("run123")
    assert status["enabled"] is True
    assert "projects" in str(status["dashboard_url"])
    assert "careeragent-ai" in str(status["dashboard_url"])


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
    assert any("Decision:" in line for line in out[0]["recommendation_rationale"])


def test_normalize_config_handles_malformed_nested_values():
    cfg = _normalize_config({"notifications": "bad", "work_modes": "remote", "geo_preferences": []})
    assert isinstance(cfg["notifications"], dict)
    assert cfg["notifications"]["email"] == ""
    assert cfg["work_modes"] == ["remote", "hybrid", "onsite"]
    assert cfg["geo_preferences"] == {"remote": True, "locations": []}


def test_normalize_config_defaults_include_source_and_role_filters():
    cfg = _normalize_config({})
    assert "linkedin.com" in cfg["allowed_job_domains"]
    assert "indeed.com" in cfg["allowed_job_domains"]
    assert cfg["role_relevance_min"] == 0.2


def test_action_token_idempotency_helpers():
    state = {}
    token = "tok_1"
    assert _is_duplicate_action(state, token) is False
    _mark_action_processed(state, token)
    assert _is_duplicate_action(state, token) is True


def test_stub_leads_use_openable_search_urls_and_demo_flag():
    leads = _stub_leads({"skills": ["AI"]}, max_jobs=3)
    assert len(leads) == 3
    assert all(x.get("is_demo") is True for x in leads)
    assert any("/jobs/search" in str(x.get("url")) for x in leads)
    assert any("indeed.com/jobs" in str(x.get("url")) for x in leads)
