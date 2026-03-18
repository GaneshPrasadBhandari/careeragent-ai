import importlib
import io
import json
import sys
import types
import asyncio
from pathlib import Path


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
_dedupe_jobs = api_main._dedupe_jobs
_record_feedback_event = api_main._record_feedback_event
_storage_paths = api_main._storage_paths
_is_dev_request_authorized = api_main._is_dev_request_authorized
_is_duplicate_action = api_main._is_duplicate_action
_mark_action_processed = api_main._mark_action_processed

_apply_role_relevance_filter = api_main._apply_role_relevance_filter
_normalize_company_name = api_main._normalize_company_name
_build_resume_markdown = api_main._build_resume_markdown
_detect_submission_success_signal = api_main._detect_submission_success_signal
_build_identity_bundle = api_main._build_identity_bundle


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


def test_storage_paths_include_feedback_partitioning_by_date_and_runid():
    paths = _storage_paths("run_abc")
    assert paths["uploads_dir"].endswith("uploads")
    assert paths["artifacts_dir"].endswith("artifacts/run_abc")
    assert "/feedback/" in paths["feedback_dir"] and paths["feedback_dir"].endswith("/run_abc")
    assert paths["logs_feedback_file"].endswith("feedback_run_abc.jsonl")
    assert paths["tracking_db"].endswith("careeragent_tracking.db")


def test_dev_storage_authorization_requires_matching_token(monkeypatch):
    monkeypatch.delenv("CAREERAGENT_DEV_TOKEN", raising=False)
    assert _is_dev_request_authorized("anything") is False

    monkeypatch.setenv("CAREERAGENT_DEV_TOKEN", "secret123")
    assert _is_dev_request_authorized("") is False
    assert _is_dev_request_authorized("wrong") is False
    assert _is_dev_request_authorized("secret123") is True


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


def test_stub_leads_cover_top_sources_with_openable_urls_and_demo_flag():
    leads = _stub_leads({"skills": ["AI"]}, max_jobs=8)
    assert len(leads) == 8
    assert all(x.get("is_demo") is True for x in leads)
    urls = [str(x.get("url")) for x in leads]
    assert any("linkedin.com/jobs/search" in u for u in urls)
    assert any("indeed.com/jobs" in u for u in urls)
    assert any("glassdoor.com" in u for u in urls)
    assert any("myvisajobs.com" in u for u in urls)


def test_dedupe_jobs_removes_exact_url_duplicates():
    jobs = [
        {"title": "AI Engineer", "company": "Acme", "url": "https://example.com/jobs/1"},
        {"title": "AI Engineer", "company": "Acme", "url": "https://example.com/jobs/1/"},
        {"title": "ML Engineer", "company": "Beta", "url": "https://example.com/jobs/2"},
    ]
    out = _dedupe_jobs(jobs)
    assert len(out) == 2


def test_role_relevance_filter_returns_best_aligned_subset_when_sparse():
    jobs = [
        {"title": f"Software Engineer {i}", "description": "backend distributed systems"}
        for i in range(80)
    ]
    cfg = {"target_roles": ["L0→L9 Planner-Director Pipeline"], "role_relevance_min": 0.2}

    out = _apply_role_relevance_filter(jobs, cfg)

    assert len(out) == 28
    assert all("role_relevance" in j for j in out)


def test_role_relevance_filter_stays_strict_when_coverage_is_healthy():
    jobs = [
        {"title": f"AI Engineer {i}", "description": "ai engineer role"} if i < 30 else {"title": f"Other Role {i}", "description": "non matching"}
        for i in range(80)
    ]
    cfg = {"target_roles": ["AI Engineer"], "role_relevance_min": 0.5}

    out = _apply_role_relevance_filter(jobs, cfg)

    assert 20 <= len(out) < len(jobs)
    assert all(float(j.get("role_relevance") or 0.0) >= 0.5 for j in out)


def test_parse_resume_timeout_uses_fallback_text(monkeypatch):
    async def _slow_to_thread(fn, *args, **kwargs):
        await asyncio.sleep(1.1)
        return fn(*args, **kwargs)

    monkeypatch.setattr(api_main.asyncio, "to_thread", _slow_to_thread)
    monkeypatch.setattr(api_main, "_fallback_resume_text", lambda _p: "Jane Doe\njane@example.com\nPython")

    profile = asyncio.run(api_main._parse_resume(Path("resume.docx"), timeout_s=0.01))

    assert profile["name"] == "Jane Doe"
    assert profile.get("parse_warning") == "resume_parse_timeout_1s"


def test_parse_resume_error_uses_fallback_text(monkeypatch):
    def _boom(_path):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(api_main, "_parse_resume_sync", _boom)
    monkeypatch.setattr(api_main, "_fallback_resume_text", lambda _p: "Alex\nalex@example.com\nML")

    profile = asyncio.run(api_main._parse_resume(Path("resume.pdf"), timeout_s=5))

    assert profile["name"] == "Alex"
    assert profile.get("parse_warning", "").startswith("resume_parse_error:RuntimeError")


def test_normalize_company_name_uses_ats_tenant_slug() -> None:
    company = _normalize_company_name(
        "boards.greenhouse.io",
        "https://boards.greenhouse.io/openai/jobs/1234",
        "Senior AI Engineer",
    )
    assert company == "Openai"


def test_resume_markdown_adds_more_projects_for_senior_profiles() -> None:
    profile = {
        "name": "Senior Candidate",
        "experience": [{"title": "Architect", "years": 12}],
        "projects": ["Project A", "Project B", "Project C", "Project D"],
        "skills": ["Python", "LLM"],
    }
    resume_md = _build_resume_markdown(profile, keyword_hints=["rag"])
    assert "### Project 3" in resume_md
    assert "### Project 4" in resume_md


def test_detect_submission_success_signal_prefers_explicit_success_markers() -> None:
    sig = _detect_submission_success_signal(
        "https://company.example.com/apply/complete?id=123",
        {"description": "Thanks for applying to this role."},
    )
    assert sig["success_state"] is True
    assert sig["signals"]


def test_identity_bundle_contains_email_targets_for_main_hidden_and_iframe() -> None:
    bundle = _build_identity_bundle(
        {"name": "Alex", "email": "alex@example.com", "phone": "+15085550123"},
        {"email": "", "phone": ""},
    )
    selectors = [x.get("selector") for x in bundle.get("email_field_targets") or []]
    assert "input[type='email']" in selectors
    assert any("hidden" in str(s) for s in selectors)
    assert any("iframe" in str(s) for s in selectors)



def test_pipeline_reaches_hitl_gate_with_discovery_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_main, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(api_main, "ARTIFACTS_DIR", tmp_path / "artifacts")
    api_main.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    api_main.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    api_main.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    api_main._runs.clear()

    async def _exercise():
        background = api_main.BackgroundTasks()

        class _ResumeUpload:
            filename = "resume.txt"

            async def read(self):
                return b"Python FastAPI LangGraph ML engineer resume"

        resume = _ResumeUpload()
        response = await api_main.start_hunt(
            background,
            resume=resume,
            hunt_config=json.dumps({"target_roles": ["AI Engineer"], "max_jobs": 5}),
        )
        run_id = response["run_id"]
        for _ in range(30):
            state = api_main._runs.get(run_id) or api_main._load_state(run_id)
            if state.get("status") in {"pending_human_input", "needs_human_approval", "completed", "error"}:
                return state
            await asyncio.sleep(0.2)
        return api_main._runs.get(run_id) or api_main._load_state(run_id)

    state = asyncio.run(_exercise())
    assert state["status"] in {"pending_human_input", "needs_human_approval", "completed"}
    assert "target_job_count" not in " ".join(state.get("errors") or [])
    assert state.get("jobs_discovered", 0) >= 5
    assert state["layers"][3]["status"] == "ok"
