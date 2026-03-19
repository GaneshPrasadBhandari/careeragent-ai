from app.ui.mission_control import (
    JobURLManager,
    _preferred_active_section,
    normalize_api_base,
    resolve_default_api_base,
)


def test_normalize_api_base_fixes_missing_scheme() -> None:
    assert normalize_api_base("demo.onrender.com") == "https://demo.onrender.com"


def test_normalize_api_base_rewrites_tips_scheme() -> None:
    assert normalize_api_base("tips://demo.onrender.com/") == "https://demo.onrender.com"


def test_resolve_default_api_base_is_locked_to_render_backend(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "https://some-other-host.example.com")
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "phase6-ui.onrender.com")
    assert resolve_default_api_base() == "https://careeragent-api.onrender.com"


def test_job_url_manager_removes_tracking_and_redirects() -> None:
    raw = "https://jobs.example.com/redirect?url=https%3A%2F%2Fwww.linkedin.com%2Fjobs%2Fview%2F123%3Futm_source%3Dfoo&trk=public_jobs"
    assert JobURLManager.sanitize(raw) == "https://www.linkedin.com/jobs/view/123"


def test_preferred_active_section_switches_to_pipeline_for_hitl() -> None:
    assert _preferred_active_section({"pending_action": "approve_ranking"}) == "📋 Pipeline Layers"
    assert _preferred_active_section({"pending_action": ""}) == "🧾 Executive Summary"
