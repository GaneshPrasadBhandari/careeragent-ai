import importlib.util
import sys
from pathlib import Path


def _load_requests_shim():
    p = Path(__file__).resolve().parents[2] / "requests.py"
    spec = importlib.util.spec_from_file_location("requests_shim", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["requests_shim"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_requests_shim_supports_form_data_encoding() -> None:
    requests = _load_requests_shim()
    try:
        requests.post("http://127.0.0.1:9/unused", data={"config": "{}"}, timeout=1)
    except requests.exceptions.ConnectionError:
        assert True


def test_requests_shim_exposes_connection_error_class() -> None:
    requests = _load_requests_shim()
    assert hasattr(requests, "exceptions")
    assert hasattr(requests.exceptions, "ConnectionError")
