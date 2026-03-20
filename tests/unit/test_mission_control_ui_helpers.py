from app.ui.mission_control import (
    JobURLManager,
    _preferred_active_section,
    build_top_portal_links,
    normalize_api_base,
    preferred_job_url,
    resolve_default_api_base,
)


def test_normalize_api_base_fixes_missing_scheme() -> None:
    assert normalize_api_base("demo.onrender.com") == "https://demo.onrender.com"
    assert normalize_api_base("127.0.0.1:10000") == "http://127.0.0.1:10000"


def test_normalize_api_base_rewrites_tips_scheme() -> None:
    assert normalize_api_base("tips://demo.onrender.com/") == "https://demo.onrender.com"


def test_resolve_default_api_base_prefers_configured_api_env(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "127.0.0.1:10000")
    assert resolve_default_api_base() == "http://127.0.0.1:10000"


def test_resolve_default_api_base_uses_reachable_local_backend(monkeypatch) -> None:
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("PUBLIC_API_URL", raising=False)
    monkeypatch.setattr("app.ui.mission_control._api_health", lambda url: url == "http://127.0.0.1:10000")
    assert resolve_default_api_base() == "http://127.0.0.1:10000"


def test_job_url_manager_removes_tracking_and_redirects() -> None:
    raw = "https://jobs.example.com/redirect?url=https%3A%2F%2Fwww.linkedin.com%2Fjobs%2Fview%2F123%3Futm_source%3Dfoo&trk=public_jobs"
    assert JobURLManager.sanitize(raw) == "https://linkedin.com/jobs/view/123"


def test_preferred_active_section_defaults_to_executive_summary_even_for_hitl() -> None:
    assert _preferred_active_section({"pending_action": "approve_ranking"}) == "🧾 Executive Summary"
    assert _preferred_active_section({"pending_action": ""}) == "🧾 Executive Summary"


def test_job_url_manager_adds_https_for_www_links() -> None:
    assert JobURLManager.sanitize("www.linkedin.com/jobs/view/123?utm_source=foo") == "https://linkedin.com/jobs/view/123"


def test_preferred_job_url_falls_back_to_redirect_and_application_urls() -> None:
    assert preferred_job_url({"redirect_url": "https://jobs.example.com/redirect?url=https%3A%2F%2Fboards.greenhouse.io%2Facme%2Fjobs%2F123"}) == "https://boards.greenhouse.io/acme/jobs/123"
    assert preferred_job_url({"application_url": "www.workday.com/company/job/456?utm_source=test"}) == "https://workday.com/company/job/456"


def test_build_top_portal_links_returns_top_8_search_portals() -> None:
    links = build_top_portal_links("AI Engineer", "Boston, MA", remote=True)
    labels = [label for label, _ in links]
    assert labels == ["LinkedIn", "Indeed", "Glassdoor", "ZipRecruiter", "Dice", "Monster", "Greenhouse", "Lever"]
    assert len({url for _, url in links}) == 8
    assert any("glassdoor.com/Job/jobs.htm" in url for _, url in links)
    assert any("indeed.com/jobs" in url for _, url in links)


def test_api_action_retries_after_503_and_succeeds(monkeypatch) -> None:
    import app.ui.mission_control as mission_control

    calls = []

    class _Resp:
        def __init__(self, status_code: int, text: str = ""):
            self.status_code = status_code
            self.text = text

    def _post(url, json, timeout):
        calls.append((url, json, timeout))
        return _Resp(503, "temporary unavailable") if len(calls) == 1 else _Resp(200, "ok")

    monkeypatch.setattr(mission_control.requests, "post", _post)
    monkeypatch.setattr(mission_control.time, "sleep", lambda *_: None)

    assert mission_control._api_action("https://api.example.com", "run-123", "approve_ranking", {"selected_job_ids": ["job-1"]}) is True
    assert len(calls) == 2


def test_api_action_treats_503_as_success_when_backend_state_already_advanced(monkeypatch) -> None:
    import app.ui.mission_control as mission_control

    class _Resp:
        def __init__(self, status_code: int, text: str = ""):
            self.status_code = status_code
            self.text = text

    monkeypatch.setattr(mission_control.requests, "post", lambda *args, **kwargs: _Resp(503, "temporary unavailable"))
    monkeypatch.setattr(mission_control.time, "sleep", lambda *_: None)
    monkeypatch.setattr(mission_control, "_api_get_status", lambda *args, **kwargs: {"status": "running", "pending_action": None})

    warnings = []
    monkeypatch.setattr(mission_control.st, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(mission_control.st, "error", lambda message: (_ for _ in ()).throw(AssertionError(message)))

    assert mission_control._api_action("https://api.example.com", "run-123", "approve_ranking", {"selected_job_ids": ["job-1"]}) is True
    assert warnings
