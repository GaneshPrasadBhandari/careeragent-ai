from app.ui.mission_control import normalize_api_base, resolve_default_api_base


def test_normalize_api_base_fixes_missing_scheme() -> None:
    assert normalize_api_base("demo.onrender.com") == "https://demo.onrender.com"


def test_normalize_api_base_rewrites_tips_scheme() -> None:
    assert normalize_api_base("tips://demo.onrender.com/") == "https://demo.onrender.com"


def test_resolve_default_api_base_is_locked_to_render_backend(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "https://some-other-host.example.com")
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "phase6-ui.onrender.com")
    assert resolve_default_api_base() == "https://careeragent-api.onrender.com"
