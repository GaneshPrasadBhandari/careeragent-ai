from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_spec = spec_from_file_location("check_api_module", Path(__file__).resolve().parents[2] / "check_api.py")
assert _spec and _spec.loader
check_api = module_from_spec(_spec)
_spec.loader.exec_module(check_api)


def test_warn_payload_marks_warning() -> None:
    out = check_api._warn("missing_key", "not set", action="set env")
    assert out["ok"] is None
    assert out["status"] == "missing_key"
    assert out["warning"] == "not set"
    assert out["action"] == "set env"


def test_network_blocked_error_detection() -> None:
    exc = RuntimeError("Tunnel connection failed: 403 Forbidden")
    assert check_api._is_network_blocked_error(exc) is True


def test_network_non_blocked_error_detection() -> None:
    exc = RuntimeError("invalid auth token")
    assert check_api._is_network_blocked_error(exc) is False
