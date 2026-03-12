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
