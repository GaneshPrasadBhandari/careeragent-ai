import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_spec = spec_from_file_location("check_env_module", Path(__file__).resolve().parents[2] / "check_env.py")
assert _spec and _spec.loader
check_env = module_from_spec(_spec)
sys.modules[_spec.name] = check_env
_spec.loader.exec_module(check_env)


def test_state_from_http_ok() -> None:
    state, detail = check_env._state_from_http(200, ok={200}, warn={400, 401})
    assert state == "active"
    assert detail == "HTTP 200"


def test_state_from_http_warning_http() -> None:
    state, detail = check_env._state_from_http(401, ok={200}, warn={400, 401})
    assert state == "warning"
    assert detail == "HTTP 401"


def test_state_from_http_network_warning() -> None:
    state, detail = check_env._state_from_http(None, ok={200}, warn={401}, err="timeout")
    assert state == "warning"
    assert "timeout" in detail


def test_state_from_http_failed_http() -> None:
    state, detail = check_env._state_from_http(418, ok={200}, warn={401})
    assert state == "failed"
    assert detail == "HTTP 418"


def test_normalize_qdrant_url_adds_scheme_and_strips_dashboard_path() -> None:
    normalized = check_env._normalize_qdrant_url("my-cluster.example.com/dashboard")
    assert normalized == "https://my-cluster.example.com"


def test_normalize_qdrant_url_keeps_custom_api_base_path() -> None:
    normalized = check_env._normalize_qdrant_url("https://gateway.example.com/qdrant")
    assert normalized == "https://gateway.example.com/qdrant"


def test_probe_qdrant_http_maps_404_to_failed(monkeypatch) -> None:
    calls = []

    def fake_get(url, headers=None, timeout=8):
        calls.append(url)
        return 404, ""

    monkeypatch.setattr(check_env, "_http_get", fake_get)
    state, detail = check_env._probe_qdrant_http("https://my-cluster.example.com", "k")
    assert state == "failed"
    assert "HTTP 404" in detail
    assert calls == [
        "https://my-cluster.example.com/collections",
        "https://my-cluster.example.com/v1/collections",
    ]


def test_probe_qdrant_http_avoids_duplicate_v1_probe(monkeypatch) -> None:
    calls = []

    def fake_get(url, headers=None, timeout=8):
        calls.append(url)
        return 404, ""

    monkeypatch.setattr(check_env, "_http_get", fake_get)
    check_env._probe_qdrant_http("https://my-cluster.example.com/v1", "k")
    assert calls == ["https://my-cluster.example.com/v1/collections"]


def test_key_state_non_strict_missing_is_warning() -> None:
    assert check_env._key_state("", strict=False) == "warning"


def test_key_state_strict_missing_is_failed() -> None:
    assert check_env._key_state("", strict=True) == "failed"


def test_key_state_present_is_active() -> None:
    assert check_env._key_state("sk-test", strict=True) == "active"
